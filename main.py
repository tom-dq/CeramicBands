import itertools
import os
import random
import time
import typing
import enum
import collections
import datetime
import contextlib
from PIL import Image


import st7
import pathlib
import math
import shutil

import config

from averaging import Averaging, AveInRadius, NoAve
from common_types import XY, ElemVectorDict, T_ResultDict, InitialSetupModelData, TEMP_ELEMS_OF_INTEREST
from relaxation import Relaxation, NoRelax
from scaling import Scaling, SingleHoleCentre, SpacedStepScaling
from tables import Table
from throttle import Throttler, StoppingCriterion, Shape, ElemPreStrainChangeData, BaseThrottler, RelaxedIncreaseDecrease
import history

import directories
import state_tracker

# To make reproducible
random.seed(123)

DONT_MAKE_MODEL_WINDOW = False

LOAD_CASE_BENDING = 1
FREEDOM_CASE = 1
STAGE = 0
TABLE_BENDING_ID = 5

STRESS_START = 400

NUM_PLATE_RES_RETRIES = 50

#fn_st7_base = pathlib.Path(r"E:\Simulations\CeramicBands\v3\Test 9C-Contact-SD2.st7")
#fn_working_image_base = pathlib.Path(r"E:\Simulations\CeramicBands\v3-pics")

#screenshot_res = st7.CanvasSize(1920, 1200)

class Actuator(enum.Enum):
    """The value used to ratchet up the prestrain."""
    S11 = enum.auto()
    SvM = enum.auto()
    s_XX = enum.auto()
    s_local = enum.auto()
    e_local = enum.auto()

    def nice_name(self) -> str:
        if self == Actuator.S11:
            return "Principal 11 Stress"

        elif self == Actuator.SvM:
            return "vM Stress"

        elif self == Actuator.s_XX:
            return "XX Global Stress"

        elif self == Actuator.s_XX:
            return "Local Directional Stress"

        elif self == Actuator.e_local:
            return "Local Directional Strain"

        else:
            raise ValueError(self)

    @property
    def input_result(self) -> st7.PlateResultType:
        if self in (Actuator.S11, Actuator.SvM, Actuator.s_XX, Actuator.s_local):
            return st7.PlateResultType.rtPlateStress

        elif self == Actuator.e_local:
            return st7.PlateResultType.rtPlateStrain

        else:
            raise ValueError(self)

class RunParams(typing.NamedTuple):
    actuator: Actuator
    stress_end: float
    scaling: Scaling
    averaging: Averaging
    relaxation: Relaxation
    throttler: BaseThrottler
    dilation_ratio: float
    n_steps_major: int
    elem_ratio_per_iter: float
    existing_prestrain_priority_factor: float

    def summary_strings(self) -> typing.Iterable[str]:
        yield "RunParams:\n"
        for field_name, field_type in self._field_types.items():
            field_val = getattr(self, field_name)
            if field_type in (Actuator,):
                output_str = field_val.nice_name()

            else:
                output_str = str(field_val)

            yield f"{field_name}\t{output_str}\n"

        yield "\n"


class Ratchet:
    """Ratchets up a table, keeping track of which element is up to where."""
    table: Table = None
    _relaxation: Relaxation
    scaling: Scaling
    _averaging: Averaging
    throttler: BaseThrottler
    min_y_so_far: dict
    max_stress_ever: float

    def __init__(self, table: Table, scaling: Scaling, relaxation: Relaxation, averaging: Averaging, throttler: BaseThrottler):
        self.table = table
        self._relaxation = relaxation
        self.scaling = scaling
        self._averaging = averaging
        self.throttler = throttler
        self.min_y_so_far = dict()

    def __copy__(self) -> "Ratchet":
        """Need to be able to make a copy of this so we can keep the state at the start of a major increment."""

        working_copy = Ratchet(
            table=self.table,
            scaling=self.scaling,
            relaxation=self._relaxation,
            averaging=self._averaging,
            throttler=self.throttler,
        )

        working_copy.min_y_so_far = self.min_y_so_far.copy()
        working_copy.max_stress_ever = self.max_stress_ever

        return working_copy

    def copy(self) -> "Ratchet":
        return self.__copy__()

    def get_all_proposed_values(self, elem_results: ElemVectorDict) -> ElemVectorDict:
        """Ratchet up the table and return the value. This is done independently for each axis.
        If 'lock_in' is set, the ratchet is "moved up". Otherwise, it's just like peeking at what the value would have been."""

        idx_to_elem_to_unaveraged = collections.defaultdict(dict)

        if TEMP_ELEMS_OF_INTEREST:
            print("Unaveraged:")

        for elem_id, idx, result_strain_raw in elem_results.as_single_values():

            # Apply scaling
            scale_key = (elem_id, idx)
            stress_scaled = self.scaling.get_x_scale_factor(scale_key) * result_strain_raw

            # Do the stress-to-prestrain lookup
            strain_raw = self.table.interp(stress_scaled)

            # Apply relaxation
            strain_relaxed = self._relaxation.relaxed(scale_key, strain_raw)

            # The unaveraged results take the ratchet value (so any locked in pre-strain is included in the averaging).
            strain_relaxed_ratcheted = self.update_minimum(False, scale_key, strain_relaxed)

            if TEMP_ELEMS_OF_INTEREST:
                if elem_id in TEMP_ELEMS_OF_INTEREST and idx == 0:
                    print(elem_id, result_strain_raw, strain_raw, strain_relaxed, strain_relaxed_ratcheted, sep='\t')


            # Save the unaveraged results
            idx_to_elem_to_unaveraged[idx][elem_id] = strain_relaxed_ratcheted

        # Perform averaging on a per-index basis (so the x-prestrain is only averaged with other x-prestrain.)
        single_vals = []

        if TEMP_ELEMS_OF_INTEREST:
            print("Averaged:")

        LOCK_IN = False
        for idx, unaveraged in idx_to_elem_to_unaveraged.items():
            averaged = self._averaging.average_results(unaveraged)

            # Do the ratcheting
            for elem_id, prestrain_val in averaged.items():
                ratchet_key = (elem_id, idx)

                ratchet_value = self.update_minimum(LOCK_IN, ratchet_key, prestrain_val)
                single_vals.append( (elem_id, idx, ratchet_value))

                if TEMP_ELEMS_OF_INTEREST:
                    if elem_id in TEMP_ELEMS_OF_INTEREST and idx == 0:
                        print(elem_id, prestrain_val, ratchet_value, sep='\t')

        # self.min_y_so_far is now updated - compose that back into the return results.
        return ElemVectorDict.from_single_values(False, single_vals)

    def update_minimum(self, lock_in: bool, scale_key, y_val) -> float:
        maybe_old_min = self.min_y_so_far.get(scale_key, math.inf)
        this_min = min(y_val, maybe_old_min)

        if lock_in:
            self.min_y_so_far[scale_key] = this_min

        return this_min

    def status_update(self) -> str:
        """Status update string"""

        val_key = {(val, key) for key,val in self.min_y_so_far.items()}
        if not val_key:
            max_val, max_elem = "...", "..."

        else:
            max_val, max_elem = min(val_key)

        non_zero = [key for key,val in self.min_y_so_far.items() if val != 0.0]

        try:
            proportion_non_zero = len(non_zero) / len(self.min_y_so_far)

        except ZeroDivisionError:
            proportion_non_zero = 0.0

        return f"Elem {max_elem}\tStrain {max_val}\tNonzero {proportion_non_zero:.3%}"


class PrestrainUpdate(typing.NamedTuple):
    elem_prestrains_locked_in: ElemVectorDict
    elem_prestrains_iteration_set: ElemVectorDict
    updated_this_round: int
    not_updated_this_round: int
    prestrained_overall: int
    update_ratio: float
    this_update_time: datetime.timedelta
    update_completed_at_time: datetime.datetime

    @staticmethod
    def zero() -> "PrestrainUpdate":
        return PrestrainUpdate(
            elem_prestrains_locked_in=ElemVectorDict(),
            elem_prestrains_iteration_set=ElemVectorDict(),
            updated_this_round=0,
            not_updated_this_round=0,
            prestrained_overall=0,
            update_ratio=0.0,
            this_update_time=datetime.timedelta(seconds=0),
            update_completed_at_time=datetime.datetime.now(),
        )

    def locked_in_prestrains(self) -> "PrestrainUpdate":
        return self._replace(elem_prestrains_locked_in=self.elem_prestrains_iteration_set)


def compress_png(png_fn):
    if not DONT_MAKE_MODEL_WINDOW:
        image = Image.open(png_fn)
        image.save(png_fn, optimize=True, quality=95)


def apply_prestrain(model: st7.St7Model, case_num: int, elem_to_ratio: typing.Dict[int, st7.Vector3]):
    """Apply all the prestrains"""

    for plate_num, prestrain_val in elem_to_ratio.items():
        prestrain = st7.Vector3(x=prestrain_val.x, y=prestrain_val.y, z=prestrain_val.z)
        model.St7SetPlatePreLoad3(plate_num, case_num, st7.PreLoadType.plPlatePreStrain, prestrain)


def setup_model_window(model_window: st7.St7ModelWindow, case_num: int):
    model_window.St7SetPlateResultDisplay_None()
    model_window.St7SetWindowResultCase(case_num)
    model_window.St7SetEntityContourIndex(st7.Entity.tyPLATE, st7.PlateContour.ctPlatePreStrainMagnitude)
    model_window.St7SetDisplacementScale(5.0, st7.ScaleType.dsAbsolute)
    model_window.St7RedrawModel(True)


def write_out_screenshot(model_window: st7.St7ModelWindow, current_result_frame: "ResultFrame"):
    setup_model_window(model_window, current_result_frame.result_case_num)
    model_window.St7ExportImage(current_result_frame.image_file, st7.ImageType.itPNG, config.active_config.screenshot_res.width, config.active_config.screenshot_res.height)
    compress_png(current_result_frame.image_file)


def write_out_to_db(db: history.DB, init_data: InitialSetupModelData, step_num_major, step_num_minor, results: st7.St7Results, current_result_frame: "ResultFrame", prestrain_update: PrestrainUpdate):

    if not config.active_config.record_result_history_in_db:
        return

    # Main case data
    db_res_case = history.ResultCase(
        num=None,
        name=str(current_result_frame),
        major_inc=step_num_major,
        minor_inc=step_num_minor,
    )

    db_case_num = db.add(db_res_case)

    # Deformed positions
    deformed_pos = get_node_positions_deformed(init_data.node_xyz, results, current_result_frame.result_case_num)
    db_node_xyz = (history.NodePosition(
        result_case_num=db_case_num,
        node_num=node_num,
        x=pos.x,
        y=pos.y,
        z=pos.z) for node_num, pos in deformed_pos.items())

    db.add_many(db_node_xyz)

    # Prestrains
    def make_prestrain_rows():
        for plate_num, pre_strain in prestrain_update.elem_prestrains_iteration_set.items():
            pre_strain_mag = (pre_strain.x**2 + pre_strain.y**2)**0.5
            yield history.ContourValue(
                result_case_num=db_case_num,
                contour_key_num=history.ContourKey.prestrain_mag.value,
                elem_num=plate_num,
                value=pre_strain_mag
            )

    db.add_many(make_prestrain_rows())


def get_results(phase_change_actuator: Actuator, results: st7.St7Results, case_num: int) -> ElemVectorDict:
    """Get the results from the result file which will be used to re-apply the prestrain."""

    res_type = phase_change_actuator.input_result

    if phase_change_actuator == Actuator.S11:
        res_sub_type = st7.PlateResultSubType.stPlateCombined
        index_list = [
            st7.St7API.ipPlateCombPrincipal11,
            st7.St7API.ipPlateCombPrincipal11,
            st7.St7API.ipPlateCombPrincipal11,
        ]

    elif phase_change_actuator == Actuator.SvM:
        res_sub_type = st7.PlateResultSubType.stPlateCombined
        index_list = [
            st7.St7API.ipPlateCombVonMises,
            st7.St7API.ipPlateCombVonMises,
            st7.St7API.ipPlateCombVonMises,
        ]

    elif phase_change_actuator == Actuator.s_XX:
        res_sub_type = st7.PlateResultSubType.stPlateGlobal
        index_list = [
            st7.St7API.ipPlateGlobalXX,
            st7.St7API.ipPlateGlobalXX,
            st7.St7API.ipPlateGlobalXX,
        ]

    elif phase_change_actuator in (Actuator.s_local, Actuator.e_local):
        res_sub_type = st7.PlateResultSubType.stPlateLocal
        index_list = [
            st7.St7API.ipPlateLocalxx,
            st7.St7API.ipPlateLocalyy,
            st7.St7API.ipPlateLocalzz,
        ]

    else:
        raise ValueError(phase_change_actuator)

    def one_plate_result(plate_num):
        worked = False
        num_tries = 0
        while num_tries < NUM_PLATE_RES_RETRIES and not worked:
            try:
                res_array = results.St7GetPlateResultArray(
                    res_type,
                    res_sub_type,
                    plate_num,
                    case_num,
                    st7.SampleLocation.spCentroid,
                    st7.PlateSurface.psPlateMidPlane,
                    0,
                )
                worked = True

            except Exception as e:
                if num_tries > 0:
                    time.sleep(0.001 * num_tries**2)

                num_tries += 1
                print(f"Failed with {e}, try {num_tries}/{NUM_PLATE_RES_RETRIES}")

                if num_tries == NUM_PLATE_RES_RETRIES:
                    raise e

        if res_array.num_points != 1:
            raise ValueError()

        result_values = [res_array.results[index] for index in index_list]
        return st7.Vector3(*result_values)

    raw_dict = {plate_num: one_plate_result(plate_num) for plate_num in results.model.entity_numbers(st7.Entity.tyPLATE)}
    return ElemVectorDict(raw_dict)


def get_node_positions_deformed(orig_positions: T_ResultDict, results: st7.St7Results, case_num: int) -> T_ResultDict:
    """Deformed node positions"""

    def one_node_pos(node_num: int):
        node_res = results.St7GetNodeResult(st7.NodeResultType.rtNodeDisp, node_num, case_num).results
        deformation = st7.Vector3(x=node_res[0], y=node_res[1], z=node_res[2])
        return orig_positions[node_num] + deformation

    return {node_num: one_node_pos(node_num) for node_num in results.model.entity_numbers(st7.Entity.tyNODE)}


def update_to_include_prestrains(
        actuator: Actuator,
        minor_acuator_input_current_raw: ElemVectorDict,
        old_prestrain_values: ElemVectorDict
) -> ElemVectorDict:
    """Make sure we include the applied pre-strains..."""

    if actuator == Actuator.e_local:
        return ElemVectorDict({
            plate_num: one_res + old_prestrain_values.get(plate_num, st7.Vector3(0.0, 0.0, 0.0)) for
            plate_num, one_res in minor_acuator_input_current_raw.items()
        })

    else:
        return minor_acuator_input_current_raw


def incremental_element_update_list(
        init_data: InitialSetupModelData,
        run_params: RunParams,
        ratchet: Ratchet,
        previous_prestrain_update: PrestrainUpdate,
        result_strain: ElemVectorDict,
        step_num_minor: int,
) -> PrestrainUpdate:
    """Gets the subset of elements which should be "yielded", based on the stress."""

    def id_key(elem_idx_val):
        """Return elem, idx"""
        return elem_idx_val[0:2]

    def val(elem_idx_val):
        """Return abs(val)"""
        return elem_idx_val[2]

    def candidate_strains(res_dict):
        ratchet.scaling.assign_working_results(previous_prestrain_update.elem_prestrains_iteration_set)
        all_res = ratchet.get_all_proposed_values(elem_results=res_dict)
        return {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in all_res.as_single_values()}

    # Use the current stress or strain results to choose the new elements.
    minor_acuator_input_current_flat = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in result_strain.as_single_values()}

    new_prestrains_all = candidate_strains(result_strain)

    old_prestrains = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in previous_prestrain_update.elem_prestrains_iteration_set.as_single_values()}
    increased_prestrains_OLD = {key: val for key, val in new_prestrains_all.items() if abs(val) > abs(old_prestrains.get(key, 0.0))}

    proposed_prestrains_changes = [
        ElemPreStrainChangeData(
            elem_num=key[0],
            axis=key[1],
            proposed_prestrain_val=val,
            old_prestrain_val=old_prestrains.get(key, 0.0),
            result_strain_val=minor_acuator_input_current_flat[key] * ratchet.scaling.get_x_scale_factor(key),
        )
        for key, val in new_prestrains_all.items()
        if abs(val - old_prestrains.get(key, 0.0)) > config.active_config.converged_delta_prestrain
    ]

    if TEMP_ELEMS_OF_INTEREST:
        old_TEMP = {elem_idx: val for elem_idx, val in old_prestrains.items() if elem_idx[0] in TEMP_ELEMS_OF_INTEREST}
        res_TEMP = {elem_idx: val for elem_idx, val in minor_acuator_input_current_flat.items() if elem_idx[0] in TEMP_ELEMS_OF_INTEREST}
        new_TEMP = {elem_idx: val for elem_idx, val in new_prestrains_all.items() if elem_idx[0] in TEMP_ELEMS_OF_INTEREST}
        scale_facts_TEMP = {elem_idx: ratchet.scaling.get_x_scale_factor(elem_idx) for elem_idx in new_TEMP.keys()}
        increased_prestrains_TEMP = [elem_strain_inc for elem_strain_inc in proposed_prestrains_changes if elem_strain_inc.elem_num in TEMP_ELEMS_OF_INTEREST]


        all_dicts = (old_TEMP, res_TEMP, scale_facts_TEMP, new_TEMP)
        all_keys = set()
        for d in all_dicts:
            rel_keys = [k for k in d.keys() if k[1] == 0]
            all_keys.update(rel_keys)

        all_bits = ['..', 'OldPS', 'ResStrain', 'Scale', 'NewPS']
        print(*all_bits, sep='\t')
        def print_line(k):
            bits = [d.get(k, '..') for d in all_dicts]
            all_bits = [k] + bits
            print(*all_bits, sep='\t')

        for k in all_keys:
            print_line(k)

        print("increased_prestrains_TEMP:")
        for inc in increased_prestrains_TEMP:
            print(inc)

        print()

    # Get the working set of prestrain updates.
    proposed_prestrains_subset = ratchet.throttler.throttle(
        init_data,
        run_params,
        proposed_prestrains_changes,
        step_num_minor
    )

    top_n_new = {
        (elem_strain_inc_data.elem_num, elem_strain_inc_data.axis): elem_strain_inc_data.proposed_prestrain_val
        for elem_strain_inc_data in proposed_prestrains_subset }

    # top_n_new = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in all_new_sorted[0:num_allowed]}
    new_count = len(top_n_new)
    left_over_count = max(0, len(proposed_prestrains_changes) - new_count)

    # Build the new pre-strain dictionary out of old and new values.
    # old_strain_single_vals = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in minor_prev_strain.as_single_values()}
    combined_final_single_values = {**old_prestrains, **top_n_new}
    single_vals_out = [ (elem, idx, val) for (elem, idx), val in combined_final_single_values.items()]
    total_out = sum(1 for _, _, val in single_vals_out if abs(val))

    # Work out now much additional dilation has been introduced.
    extra_dilation = [
        (elem_idx, val - old_prestrains.get(elem_idx, 0.0))
        for elem_idx, val in top_n_new.items()
    ]

    extra_dilation_norm = sum(init_data.elem_volume[elem_idx[0]] * abs(val) for elem_idx, val in extra_dilation)

    update_completed_at_time = datetime.datetime.now()
    this_update_time = update_completed_at_time - previous_prestrain_update.update_completed_at_time

    return PrestrainUpdate(
        elem_prestrains_locked_in=previous_prestrain_update.elem_prestrains_locked_in,
        elem_prestrains_iteration_set=ElemVectorDict.from_single_values(True, single_vals_out),
        updated_this_round=new_count,
        not_updated_this_round=left_over_count,
        prestrained_overall=total_out,
        update_ratio=extra_dilation_norm,
        this_update_time=this_update_time,
        update_completed_at_time=update_completed_at_time,
    )



def make_fn(working_dir: pathlib.Path, actuator: Actuator, n_steps: int, scaling, averaging, stress_start, stress_end, dilation_ratio, relaxation, elem_ratio_per_iter, existing_prestrain_priority_factor) -> pathlib.Path:
    """Makes the base Strand7 model name."""
    new_stem = config.active_config.fn_st7_base.stem + f" - {actuator.name} {stress_start} to {stress_end} DilRat={dilation_ratio} Steps={n_steps} {scaling} {averaging} {relaxation} ElemRatio={elem_ratio_per_iter} ExistingPriority={existing_prestrain_priority_factor}"

    new_name = new_stem + config.active_config.fn_st7_base.suffix
    return working_dir / new_name
    #return config.fn_st7_base.with_name(new_name)


def fn_append(fn_orig: pathlib.Path, extra_bit) -> pathlib.Path:
    new_stem = f"{fn_orig.stem} - {extra_bit}"
    new_name = new_stem + fn_orig.suffix
    return fn_orig.with_name(new_name)


class ResultFrame(typing.NamedTuple):
    """Encapsulates the data associated with a single result case, in a specified result file."""
    st7_file: pathlib.Path
    configuration: config.Config
    result_file_index: int
    result_case_num: int  # Result case number in the current file - this may be reused.
    global_result_case_num: int   # Case number which is unique across all files (if result files are split).
    load_time_table: typing.Optional[Table]
    prev_result_case: typing.Optional["ResultFrame"] # Store this with the full chain of history - just go one step back.

    def __str__(self):
        """Trimmed down version..."""
        return f"ResultFrame(result_file_index={self.result_file_index}, result_case_num={self.result_case_num}, global_result_case_num={self.global_result_case_num})"

    def delete_old_files_if_needed(self):
        if config.active_config.delete_old_result_files:
            two_files_ago = self._replace(result_file_index=self.result_file_index-2)
            old_res_file = two_files_ago.result_file
            try:
                os.remove(str(old_res_file))

            except FileNotFoundError:
                pass


    def get_next_result_frame(self, bending_load_factor: float, advance_result_case: bool) -> "ResultFrame":

        if advance_result_case:
            proposed_new_result_case_num = self.result_case_num + 1

        else:
            proposed_new_result_case_num = self.result_case_num

        this_case_with_no_history = self._replace(prev_result_case=None)

        if self.configuration.solver == st7.SolverType.stNonlinearStatic:
            return self._replace(
                result_case_num=proposed_new_result_case_num,
                global_result_case_num=self.global_result_case_num + 1,
                prev_result_case=this_case_with_no_history,
            )

        elif self.configuration.solver == st7.SolverType.stQuasiStatic:
            need_new_result_file = proposed_new_result_case_num > self.configuration.qsa_steps_per_file

            if need_new_result_file:
                working_new_frame = self._replace(
                    result_case_num=1,
                    result_file_index=self.result_file_index + 1,
                    global_result_case_num=self.global_result_case_num + 1,
                )

                self.delete_old_files_if_needed()

            else:
                working_new_frame = self._replace(
                    result_case_num=proposed_new_result_case_num,
                    global_result_case_num=self.global_result_case_num + 1,
                )

            # Add the new datapoint on the end. And an extra one, for the whole time step.
            final_table_datapoint_A = XY(x=self.total_time_at_step_end, y=bending_load_factor)
            final_table_datapoint_B = XY(x=self.total_time_at_step_end + self.configuration.qsa_time_step_size, y=bending_load_factor)

            new_table_working = self.load_time_table.with_appended_datapoint(final_table_datapoint_A)
            new_table = new_table_working.with_appended_datapoint(final_table_datapoint_B)
            return working_new_frame._replace(
                load_time_table=new_table,
                prev_result_case=this_case_with_no_history,
            )

        else:
            raise ValueError(self.configuration.solver)

    @property
    def result_file(self) -> pathlib.Path:
        if self.configuration.solver == st7.SolverType.stNonlinearStatic:
            return self.st7_file.with_suffix(".NLA")

        elif self.configuration.solver == st7.SolverType.stQuasiStatic:
            number_suffix = f"{self.result_file_index:04}"
            new_name = f"{self.st7_file.stem}_{number_suffix}.QSA"
            return self.st7_file.with_name(new_name)

        else:
            raise ValueError(self.configuration.solver)

    @property
    def restart_file(self) -> pathlib.Path:
        if self.configuration.solver == st7.SolverType.stNonlinearStatic:
            return self.result_file.with_suffix(".SRF")

        elif self.configuration.solver == st7.SolverType.stQuasiStatic:
            return self.result_file.with_suffix(".QRF")

        else:
            raise ValueError(self.configuration.solver)

    @property
    def working_dir(self) -> pathlib.Path:
        return self.st7_file.parent

    @property
    def image_file(self) -> pathlib.Path:
        return self.working_dir / f"Case-{self.global_result_case_num:04}.png"

    @property
    def total_time_at_step_end(self) -> float:
        return self.configuration.qsa_time_step_size * self.global_result_case_num


def add_increment(model: st7.St7Model, result_frame: ResultFrame, this_load_factor, inc_name, advance_result_case) -> ResultFrame:
    next_result_frame = result_frame.get_next_result_frame(this_load_factor, advance_result_case)

    if result_frame.configuration.solver == st7.SolverType.stNonlinearStatic:
        model.St7AddNLAIncrement(STAGE, inc_name)
        this_inc = model.St7GetNumNLAIncrements(STAGE)
        if next_result_frame.global_result_case_num != this_inc:
            raise ValueError(f"Got {this_inc} from St7GetNumNLAIncrements but {next_result_frame.global_result_case_num} from the ResultFrame...")

    elif result_frame.configuration.solver == st7.SolverType.stQuasiStatic:
        # For QSA, we roll over the result files. So sometimes this will have changed.
        model.St7SetResultFileName(next_result_frame.result_file)
        model.St7SetStaticRestartFile(next_result_frame.restart_file)

    else:
        raise ValueError("sovler type")

    return next_result_frame


def set_restart_for(model: st7.St7Model, result_frame: ResultFrame):
    previous_result_frame = result_frame.prev_result_case
    if result_frame.configuration.solver == st7.SolverType.stNonlinearStatic:
        model.St7SetNLAInitial(previous_result_frame.result_file, previous_result_frame.result_case_num)

    elif result_frame.configuration.solver == st7.SolverType.stQuasiStatic:
        model.St7SetQSAInitial(previous_result_frame.result_file, previous_result_frame.result_case_num)

    else:
        raise ValueError(result_frame.configuration.solver)


def set_load_increment_table(model: st7.St7Model, result_frame: ResultFrame, this_bending_load_factor, prestrain_load_case_num):
    # Set the load and freedom case - can use either method. Don't use both though!

    if result_frame.configuration.solver == st7.SolverType.stNonlinearStatic:
        model.St7SetNLAFreedomIncrementFactor(STAGE, result_frame.result_case_num, FREEDOM_CASE, this_bending_load_factor)
        for iLoadCase in range(1, model.St7GetNumLoadCase() + 1):
            if iLoadCase == LOAD_CASE_BENDING:
                factor = this_bending_load_factor

            elif iLoadCase == prestrain_load_case_num:
                factor = 1.0

            else:
                factor = 0.0

            model.St7SetNLALoadIncrementFactor(STAGE, result_frame.result_case_num, iLoadCase, factor)

    elif result_frame.configuration.solver == st7.SolverType.stQuasiStatic:
        # Enable / disable the load and freedom cases.
        model.St7EnableTransientFreedomCase(FREEDOM_CASE)
        for iLoadCase in range(1, model.St7GetNumLoadCase() + 1):
            should_be_enabled = iLoadCase in (LOAD_CASE_BENDING, prestrain_load_case_num)
            if should_be_enabled:
                model.St7EnableTransientLoadCase(iLoadCase)

            else:
                model.St7DisableTransientLoadCase(iLoadCase)

        # Update the tables so we get the right bending factor.
        model.St7SetTableTypeData(
            st7.TableType.ttVsTime,
            TABLE_BENDING_ID,
            len(result_frame.load_time_table.data),
            result_frame.load_time_table.as_flat_doubles(),
        )

    else:
        raise ValueError(result_frame.configuration.solver)


def set_max_iters(model: st7.St7Model, max_iters: typing.Optional[config.MaxIters], use_major:bool):
    """Put a hard out at some number of increments."""

    if max_iters:
        iter_num = max_iters.major_step if use_major else max_iters.minor_step
        model.St7SetSolverDefaultsLogical(st7.SolverDefaultLogical.spAllowExtraIterations, False)
        model.St7SetSolverDefaultsInteger(st7.SolverDefaultInteger.spMaxIterationNonlin, iter_num)


def initial_setup(model: st7.St7Model, initial_result_frame: ResultFrame) -> InitialSetupModelData:

    model.St7EnableSaveRestart()
    model.St7EnableSaveLastRestartStep()

    elem_centroid = {
        elem_num: model.St7GetElementCentroid(st7.Entity.tyPLATE, elem_num, 0)
        for elem_num in model.entity_numbers(st7.Entity.tyPLATE)
    }

    node_xyz = {node_num: model.St7GetNodeXYZ(node_num) for node_num in model.entity_numbers(st7.Entity.tyNODE)}

    elem_conns = {
        plate_num: model.St7GetElementConnection(st7.Entity.tyPLATE, plate_num) for
        plate_num in model.entity_numbers(st7.Entity.tyPLATE)
    }

    elem_volume = {
        plate_num: model.St7GetElementData(st7.Entity.tyPLATE, plate_num) for
        plate_num in model.entity_numbers(st7.Entity.tyPLATE)
    }

    total_volume = sum(elem_volume.values())
    elem_volume_ratio = {elem: vol/total_volume for elem, vol in elem_volume.items()}

    if initial_result_frame.configuration.solver == st7.SolverType.stNonlinearStatic:
        # If there's no first increment, create one.
        starting_incs = model.St7GetNumNLAIncrements(STAGE)
        if starting_incs == 0:
            model.St7AddNLAIncrement(STAGE, "Initial")

        elif starting_incs == 1:
            pass

        else:
            raise Exception("Already had increments?")

        model.St7EnableNLALoadCase(STAGE, LOAD_CASE_BENDING)
        model.St7EnableNLAFreedomCase(STAGE, FREEDOM_CASE)

    else:
        # Make the time table a single row.
        model.St7SetNumTimeStepRows(1)
        model.St7SetTimeStepData(1, 1, 1, initial_result_frame.configuration.qsa_time_step_size)
        model.St7SetSolverDefaultsLogical(st7.SolverDefaultLogical.spAppendRemainingTime, False)  # Whole table is always just one step.

        model.St7EnableTransientLoadCase(LOAD_CASE_BENDING)
        model.St7EnableTransientFreedomCase(FREEDOM_CASE)

        # Set up the tables which will drive the bending load.
        model.St7NewTableType(
            st7.TableType.ttVsTime,
            TABLE_BENDING_ID,
            len(initial_result_frame.load_time_table.data),
            "Bending Load Factor",
            initial_result_frame.load_time_table.as_flat_doubles(),
        )

        model.St7SetTransientLoadTimeTable(LOAD_CASE_BENDING, TABLE_BENDING_ID, False)
        model.St7SetTransientFreedomTimeTable(FREEDOM_CASE, TABLE_BENDING_ID, False)


    # For all solvers.
    model.St7SetResultFileName(initial_result_frame.result_file)
    model.St7SetStaticRestartFile(initial_result_frame.restart_file)


    return InitialSetupModelData(
        node_xyz=node_xyz,
        elem_centroid=elem_centroid,
        elem_conns=elem_conns,
        elem_volume=elem_volume,
        elem_volume_ratio=elem_volume_ratio,
    )


def _make_prestrain_table(run_params: RunParams) -> Table:
    if run_params.actuator.input_result == st7.PlateResultType.rtPlateStress:
        prestrain_table = Table([
            XY(0.0, 0.0),
            XY(STRESS_START, 0.0),
            XY(run_params.stress_end, -1 * run_params.dilation_ratio),
            XY(run_params.stress_end + 200, -1 * run_params.dilation_ratio),
        ])

    elif run_params.actuator.input_result == st7.PlateResultType.rtPlateStrain:
        youngs_mod = 220000  # Hacky way!
        prestrain_table = Table([
            XY(0.0, 0.0),
            XY(STRESS_START / youngs_mod, 0.0),
            XY(run_params.stress_end / youngs_mod, -1 * run_params.dilation_ratio),
            XY((run_params.stress_end + 200) / youngs_mod, -1 * run_params.dilation_ratio),
        ])

    else:
        raise ValueError(run_params.actuator.input_result)

    return prestrain_table



def main(run_params: RunParams):

    prestrain_table = _make_prestrain_table(run_params)
    ratchet = Ratchet(
        prestrain_table,
        scaling=run_params.scaling,
        averaging=run_params.averaging,
        relaxation=run_params.relaxation,
        throttler=run_params.throttler)

    # Allow a maximum of 10% of the elements to yield in a given step.
    n_steps_minor_max = math.inf

    for summary_string in run_params.summary_strings():
        print(summary_string.strip())
        
    for summary_string in config.active_config.summary_strings():
        print(summary_string.strip())

    print(f"Limiting to {n_steps_minor_max} steps per load increment - only {run_params.elem_ratio_per_iter:%} can yield.")

    working_dir = directories.get_unique_sub_dir(config.active_config.fn_working_image_base)

    fn_st7 = working_dir / "Model.st7"
    fn_db = working_dir / "history.db"

    shutil.copy2(config.active_config.fn_st7_base, fn_st7)

    current_result_frame = ResultFrame(
        st7_file=fn_st7,
        configuration=config.active_config,
        result_file_index=0,
        result_case_num=1,
        global_result_case_num=1,
        load_time_table=Table([XY(0.0, 0.0), XY(config.active_config.qsa_time_step_size, 0.0)]),
        prev_result_case=None,
    )

    with open(working_dir / "Meta.txt", "w") as f_meta:
        f_meta.writelines(run_params.summary_strings())
        f_meta.writelines(config.active_config.summary_strings())

    print(f"Working directory: {working_dir}")
    print()

    state = state_tracker.default_state
    print("Signal files:")
    state.print_signal_file_names(working_dir)

    with contextlib.ExitStack() as exit_stack:
        model = exit_stack.enter_context(st7.St7Model(fn_st7, config.active_config.scratch_dir))
        model_window = exit_stack.enter_context(model.St7CreateModelWindow(DONT_MAKE_MODEL_WINDOW))
        db = exit_stack.enter_context(history.DB(fn_db))

        init_data = initial_setup(model, current_result_frame)
        db.add_element_connections(init_data.elem_conns)

        scaling.assign_centroids(init_data)
        averaging.populate_radius(init_data.node_xyz, init_data.elem_conns)

        # Dummy init values
        prestrain_update = PrestrainUpdate.zero()

        # Assume the elements are evenly sized. Factor of 2 is for x and y.
        n_updates_per_iter = round(2 * run_params.elem_ratio_per_iter * len(model.entity_numbers(st7.Entity.tyPLATE)))

        set_max_iters(model, config.active_config.max_iters, use_major=True)
        model.St7RunSolver(current_result_frame.configuration.solver, st7.SolverMode.smBackgroundRun, True)

        previous_load_factor = 0.0

        # old_prestrain_values = ElemVectorDict()
        for step_num_major in range(run_params.n_steps_major):

            # Perform an iterative update in which only a certain proportion of the elements can "yield" at once.
            minor_step_iter = itertools.count()
            step_num_minor = next(minor_step_iter)

            def step_name():
                return f"{step_num_major + 1}.{step_num_minor}"

            # Get the results from the last major step.
            with model.open_results(current_result_frame.result_file) as results:
                write_out_screenshot(model_window, current_result_frame)
                write_out_to_db(db, init_data, step_num_major, step_num_minor, results, current_result_frame, prestrain_update)

            # Update the model with the new load
            this_load_factor = (step_num_major+1) / run_params.n_steps_major

            prestrain_load_case_num = create_load_case(model, step_name())
            apply_prestrain(model, prestrain_load_case_num, prestrain_update.elem_prestrains_locked_in)

            # Add the increment, or overwrite it
            current_result_frame = add_increment(model, current_result_frame, this_load_factor, step_name(), advance_result_case=True)
            set_load_increment_table(model, current_result_frame, this_load_factor, prestrain_load_case_num)

            # Make sure we're starting from the last case
            set_restart_for(model, current_result_frame)

            relaxation.set_current_relaxation(previous_load_factor, this_load_factor)

            step_num_minor = next(minor_step_iter)
            new_count = math.inf
            while (new_count > 0) and (step_num_minor < n_steps_minor_max):

                model.St7SaveFile()
                model.St7RunSolver(current_result_frame.configuration.solver, st7.SolverMode.smBackgroundRun, True)

                # For the next minor increment, unless overwritten.
                set_max_iters(model, config.active_config.max_iters, use_major=False)

                # Get the results from the last minor step.
                with model.open_results(current_result_frame.result_file) as results:
                    result_strain_raw = get_results(run_params.actuator, results, current_result_frame.result_case_num)
                    result_strain = update_to_include_prestrains(run_params.actuator, result_strain_raw, prestrain_update.elem_prestrains_iteration_set)
                    write_out_screenshot(model_window, current_result_frame)
                    write_out_to_db(db, init_data, step_num_major, step_num_minor, results, current_result_frame, prestrain_update)

                prestrain_update = incremental_element_update_list(
                    init_data=init_data,
                    run_params=run_params,
                    ratchet=ratchet,
                    previous_prestrain_update=prestrain_update,
                    result_strain=result_strain,
                    step_num_minor=step_num_minor,
                )

                new_count = prestrain_update.updated_this_round
                update_bits = [
                    f"{step_name()}/{run_params.n_steps_major}",
                    f"Updated {new_count}",
                    f"Left {prestrain_update.not_updated_this_round}",
                    f"Total {prestrain_update.prestrained_overall}",
                    f"Norm {prestrain_update.update_ratio}",
                    f"TimeDelta {prestrain_update.this_update_time.total_seconds():1.3f}"
                    #str(ratchet.status_update()),
                ]
                print("\t".join(update_bits))

                # Only continue if there is updating to do.
                if prestrain_update.updated_this_round > 0:
                    if config.active_config.case_for_every_increment:
                        prestrain_load_case_num = create_load_case(model, step_name())

                    # Add the next step
                    current_result_frame = add_increment(model, current_result_frame, this_load_factor, step_name(), advance_result_case=True)
                    set_load_increment_table(model, current_result_frame, this_load_factor, prestrain_load_case_num)

                    # Make sure we're starting from the last case
                    set_restart_for(model, current_result_frame)

                    apply_prestrain(model, prestrain_load_case_num, prestrain_update.elem_prestrains_iteration_set)

                    if config.active_config.ratched_prestrains_during_iterations:
                        # Update the ratchet settings for the one we did ramp up.
                        for elem, idx, val in prestrain_update.elem_prestrains_iteration_set.as_single_values():
                            ratchet.update_minimum(True, (elem, idx), val)

                    # Keep track of the old results...
                    step_num_minor = next(minor_step_iter)

                    set_max_iters(model, config.active_config.max_iters, use_major=True)

                # Update the state.
                state = state.update_from_fn(working_dir)
                if state.need_to_write_st7():
                    model.St7SaveFile()

                if state.execution == state_tracker.Execution.pause:
                    _ = input("Press enter to carry on...")
                    state = state.unpause()

                elif state.execution == state_tracker.Execution.stop:
                    return

            # Update the ratchet with the equilibrated results.
            # for elem, idx, val in prestrain_update.elem_prestrains.as_single_values():
            #     scale_key = (elem, idx)
            #    ratchet.update_minimum(True, scale_key, val)

            prestrain_update = prestrain_update.locked_in_prestrains()
            previous_load_factor = this_load_factor

            # Update the ratchet for what's been locked in.
            for elem, idx, val in prestrain_update.elem_prestrains_locked_in.as_single_values():
                ratchet.update_minimum(True, (elem, idx), val)

            relaxation.flush_previous_values()

        model.St7SaveFile()

        # Save the image of pre-strain results from the maximum load step.
        with model.open_results(current_result_frame.result_file) as results, model.St7CreateModelWindow(dont_really_make=False) as model_window:
            write_out_screenshot(model_window, current_result_frame)
            write_out_to_db(db, init_data, step_num_major, step_num_minor, results, current_result_frame, prestrain_update)


def create_load_case(model, case_name):
    model.St7NewLoadCase(case_name)
    new_case_num = model.St7GetNumLoadCase()
    model.St7EnableNLALoadCase(STAGE, new_case_num)
    return new_case_num


if __name__ == "__main__":
    dilation_ratio = 0.008   # 0.8% expansion, according to Jerome
    elem_ratio_per_iter = 0.0005

    #relaxation = LimitedIncreaseRelaxation(0.01)
    #relaxation = PropRelax(0.5)
    relaxation = NoRelax()

    scaling = SpacedStepScaling(y_depth=0.04, spacing=0.2, amplitude=0.5, hole_width=0.04, adj_strain_factor=0.0 / dilation_ratio)
    #scaling = SingleHoleCentre(y_depth=0.25, amplitude=0.2, hole_width=0.1)
    #scaling = CosineScaling(y_depth=0.25, spacing=0.4, amplitude=0.2)

    # averaging = AveInRadius(0.02)
    averaging = NoAve()

    def throttle_ratio_decay(step_num_minor):
        return (step_num_minor+1)**-0.7

    # throttler = Throttler(stopping_criterion=StoppingCriterion.volume_ratio, shape=Shape.step, cutoff_value=elem_ratio_per_iter)
    # throttler = Throttler(stopping_criterion=StoppingCriterion.new_prestrain_total, shape=Shape.linear, cutoff_value=elem_ratio_per_iter * dilation_ratio * 2)
    throttler = RelaxedIncreaseDecrease(ratio_getter=throttle_ratio_decay)

    run_params = RunParams(
        actuator=Actuator.e_local,
        stress_end=440.0,
        scaling=scaling,
        averaging=averaging,
        relaxation=relaxation,
        throttler=throttler,
        dilation_ratio=dilation_ratio,
        n_steps_major=10,
        elem_ratio_per_iter=elem_ratio_per_iter,
        existing_prestrain_priority_factor=2,
    )

    main(run_params)


# Combine to one video with "C:\Utilities\ffmpeg-20181212-32601fb-win64-static\bin\ffmpeg.exe -f image2 -r 12 -i Case-%04d.png -vcodec libx264 -profile:v high444 -refs 16 -crf 0 out.mp4"

