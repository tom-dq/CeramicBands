import itertools
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

from averaging import Averaging, AveInRadius
from common_types import XY, ElemVectorDict, T_ResultDict
from relaxation import Relaxation, PropRelax
from scaling import Scaling, SpacedStepScaling
from tables import Table
import history

import directories
import state_tracker


# To make reproducible
random.seed(123)

RATCHET_AT_INCREMENTS = True
DEBUG_CASE_FOR_EVERY_INCREMENT = False
RECORD_HISTORY = True
DONT_MAKE_MODEL_WINDOW = False

LOAD_CASE_BENDING = 1
FREEDOM_CASE = 1
STAGE = 0
TABLE_BENDING_ID = 5

STRESS_START = 400

NUM_PLATE_RES_RETRIES = 10

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


class Ratchet:
    """Ratchets up a table, keeping track of which element is up to where."""
    table: Table = None
    _relaxation: Relaxation
    scaling: Scaling
    _averaging: Averaging
    min_y_so_far: dict
    max_stress_ever: float

    def __init__(self, table: Table, scaling: Scaling, relaxation: Relaxation, averaging: Averaging):
        self.table = table
        self._relaxation = relaxation
        self.scaling = scaling
        self._averaging = averaging
        self.min_y_so_far = dict()
        self.max_stress_ever = -1 * math.inf

    def __copy__(self) -> "Ratchet":
        """Need to be able to make a copy of this so we can keep the state at the start of a major increment."""

        working_copy = Ratchet(
            table=self.table,
            scaling=self.scaling,
            relaxation=self._relaxation,
            averaging=self._averaging
        )

        working_copy.min_y_so_far = self.min_y_so_far.copy()
        working_copy.max_stress_ever = self.max_stress_ever

        return working_copy

    def copy(self) -> "Ratchet":
        return self.__copy__()

    def get_all_values(self, lock_in: bool, elem_results: ElemVectorDict) -> ElemVectorDict:
        """Ratchet up the table and return the value. This is done independently for each axis.
        If 'lock_in' is set, the ratchet is "moved up". Otherwise, it's just like peeking at what the value would have been."""

        idx_to_elem_to_unaveraged = collections.defaultdict(dict)

        for elem_id, idx, stress_raw in elem_results.as_single_values():
            if lock_in:
                self.max_stress_ever = max(self.max_stress_ever, stress_raw)

            # Apply scaling
            scale_key = (elem_id, idx)
            stress_scaled = self.scaling.get_x_scale_factor(scale_key) * stress_raw

            # Do the stress-to-prestrain lookup
            strain_raw = self.table.interp(stress_scaled)

            # Apply relaxation
            strain_relaxed = self._relaxation.relaxed(scale_key, strain_raw)

            # Save the unaveraged results
            idx_to_elem_to_unaveraged[idx][elem_id] = strain_relaxed

        # Perform averaging on a per-index basis (so the x-prestrain is only averaged with other x-prestrain.)
        single_vals = []
        for idx, unaveraged in idx_to_elem_to_unaveraged.items():
            averaged = self._averaging.average_results(unaveraged)

            # Do the ratcheting
            for elem_id, prestrain_val in averaged.items():
                ratchet_key = (elem_id, idx)

                ratchet_value = self.update_minimum(lock_in, ratchet_key, prestrain_val)
                single_vals.append( (elem_id, idx, ratchet_value))

        # self.min_y_so_far is now updated - compose that back into the return results.
        return ElemVectorDict.from_single_values(False, single_vals)

    def update_minimum(self, lock_in: bool, scale_key, y_val) -> float:
        maybe_old_min = self.min_y_so_far.get(scale_key, math.inf)
        this_min = min(y_val, maybe_old_min)

        if lock_in:
            self.min_y_so_far[scale_key] = this_min

        return this_min

    def _get_value_single(self, scale_key, x_unscaled: float) -> float:
        """Interpolates a single value."""

        self.max_stress_ever = max(self.max_stress_ever, x_unscaled)

        # Apply the scaling.
        x_scaled = self.scaling.get_x_scale_factor(scale_key) * x_unscaled

        # Look up the table value
        y_val_unrelaxed = self.table.interp(x_scaled)

        # Apply the relaxation on the looked-up value.
        y_val = self._relaxation.relaxed(scale_key, y_val_unrelaxed)

        self.update_minimum(scale_key, y_val)

        return self.min_y_so_far[scale_key]


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

        return f"Elem {max_elem}\tMaxStress {self.max_stress_ever}\tStrain {max_val}\tNonzero {proportion_non_zero:.3%}"



class InitialSetupModelData(typing.NamedTuple):
    node_xyz: typing.Dict[int, st7.Vector3]
    elem_centroid: typing.Dict[int, st7.Vector3]
    elem_conns: typing.Dict[int, typing.Tuple[int, ...]]
    elem_volume: typing.Dict[int, float]


class PrestrainUpdate(typing.NamedTuple):
    elem_prestrains: ElemVectorDict
    updated_this_round: int
    not_updated_this_round: int
    prestrained_overall: int
    update_ratio: float
    this_update_time: datetime.timedelta
    update_completed_at_time: datetime.datetime

    @staticmethod
    def zero() -> "PrestrainUpdate":
        return PrestrainUpdate(
            elem_prestrains=ElemVectorDict(),
            updated_this_round=0,
            not_updated_this_round=0,
            prestrained_overall=0,
            update_ratio=0.0,
            this_update_time=datetime.timedelta(seconds=0),
            update_completed_at_time=datetime.datetime.now(),
        )


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
        for plate_num, pre_strain in prestrain_update.elem_prestrains.items():
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

        if not worked:
            raise Exception("Ran out of chances to get plate result.")


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
        previous_prestrain_update: PrestrainUpdate,
        num_allowed: int,
        existing_prestrain_priority_factor: float,
        elem_volume: typing.Dict[int, float],
        ratchet: Ratchet,
        minor_prev_prestrain: ElemVectorDict,
        minor_acuator_input_current: ElemVectorDict,
) -> PrestrainUpdate:
    """Gets the subset of elements which should be "yielded", based on the stress."""

    def id_key(elem_idx_val):
        """Return elem, idx"""
        return elem_idx_val[0:2]

    def val(elem_idx_val):
        """Return abs(val)"""
        return elem_idx_val[2]

    def candidate_strains(res_dict):
        all_res = ratchet.get_all_values(lock_in=False, elem_results=res_dict)
        return {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in all_res.as_single_values()}

    #old_prestrains = candidate_strains(minor_acuator_input_prev)
    new_prestrains_all = candidate_strains(minor_acuator_input_current)

    old_prestrains = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in minor_prev_prestrain.as_single_values()}
    increased_prestrains = {key: val for key, val in new_prestrains_all.items() if abs(val) > abs(old_prestrains.get(key, 0.0))}

    # Use the current stress results to choose the new elements.
    minor_acuator_input_current_flat = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in minor_acuator_input_current.as_single_values()}
    #minor_current_strain_flat = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in minor_current_strain.as_single_values()}

    def priority_stress(elem_idx_val):
        """Takes in a prestrain item. This decides which prestrains get updated this round."""
        elem_idx = (elem_idx_val[0], elem_idx_val[1])
        current_stress_unscaled = minor_acuator_input_current_flat[elem_idx]

        existing_scale_ratio = abs(old_prestrains.get(elem_idx, 0.0)) / ratchet.table.max_abs_y
        scale_factor = existing_prestrain_priority_factor * existing_scale_ratio
        current_stress_scaled = current_stress_unscaled * ratchet.scaling.get_x_scale_factor(elem_idx) * scale_factor
        return abs(current_stress_scaled)

    def priority_strain(elem_idx_val):
        elem_idx = (elem_idx_val[0], elem_idx_val[1])
        current_strain_results_scaled = minor_current_strain_flat[elem_idx] * ratchet.scaling.get_x_scale_factor(elem_idx)
        current_applied_strain = existing_prestrain_priority_factor * old_prestrains.get(elem_idx, 0.0)
        return abs(current_strain_results_scaled + current_applied_strain)

    def priorty_strain_simple(elem_idx_val):
        elem_idx = (elem_idx_val[0], elem_idx_val[1])
        # Assuming a strain actuation, the input is the true actuator value...

        proposed_new_prestrain = minor_acuator_input_current_flat[elem_idx] * ratchet.scaling.get_x_scale_factor(elem_idx)

        # Previous pre-strain will be zero if the element is:
        # - zero if the element is undialated.
        # - negative if the element is dialated.
        strain_previous = old_prestrains.get(elem_idx, 0.0)

        # Give a boost to the already-strained elements
        return abs(proposed_new_prestrain) + existing_prestrain_priority_factor * abs(strain_previous)


    # Get the top N elements.
    all_new = ( (elem_idx[0], elem_idx[1], val) for elem_idx, val in increased_prestrains.items())
    all_new_sorted = sorted(all_new, key=priorty_strain_simple, reverse=True)

    top_n_new = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in all_new_sorted[0:num_allowed]}
    new_count = len(top_n_new)
    left_over_count = max(0, len(all_new_sorted) - new_count)

    # Build the new pre-strain dictionary out of old and new values.
    # old_strain_single_vals = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in minor_prev_strain.as_single_values()}
    combined_final_single_values = {**old_prestrains, **top_n_new}
    single_vals_out = [ (elem, idx, val) for (elem, idx), val in combined_final_single_values.items()]
    total_out = sum(1 for _, _, val in single_vals_out if abs(val))

    # Update the ratchet settings for the one we did ramp up.
    for elem_idx, val in top_n_new.items():
        ratchet.update_minimum(True, elem_idx, val)

    # Work out now much additional dilation has been introduced.
    extra_dilation = [
        (elem_idx, val - old_prestrains.get(elem_idx, 0.0))
        for elem_idx, val in top_n_new.items()
    ]

    extra_dilation_norm = sum(elem_volume[elem_idx[0]] * abs(val) for elem_idx, val in extra_dilation)

    update_completed_at_time = datetime.datetime.now()
    this_update_time = update_completed_at_time - previous_prestrain_update.update_completed_at_time

    return PrestrainUpdate(
        elem_prestrains=ElemVectorDict.from_single_values(True, single_vals_out),
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

    def __str__(self):
        """Trimmed down version..."""
        return f"ResultFrame(result_file_index={self.result_file_index}, result_case_num={self.result_case_num}, global_result_case_num={self.global_result_case_num})"

    def get_next_result_frame(self, bending_load_factor: float, advance_result_case: bool) -> "ResultFrame":

        if advance_result_case:
            proposed_new_result_case_num = self.result_case_num + 1

        else:
            proposed_new_result_case_num = self.result_case_num

        if self.configuration.solver == st7.SolverType.stNonlinearStatic:
            return self._replace(
                result_case_num=proposed_new_result_case_num,
                global_result_case_num=self.global_result_case_num + 1
            )

        elif self.configuration.solver == st7.SolverType.stQuasiStatic:
            need_new_result_file = proposed_new_result_case_num > self.configuration.qsa_steps_per_file

            if need_new_result_file:
                working_new_frame = self._replace(
                    result_case_num=1,
                    result_file_index=self.result_file_index + 1,
                    global_result_case_num=self.global_result_case_num + 1,
                )

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
            return working_new_frame._replace(load_time_table=new_table)

        else:
            raise ValueError(self.configuration.solver)

    def get_previous_result_frame(self) -> "ResultFrame":
        if self.global_result_case_num < 2:
            raise ValueError("This is the first one!")

        if self.configuration.solver == st7.SolverType.stNonlinearStatic:
            return self._replace(
                result_case_num=self.result_case_num - 1,
                global_result_case_num=self.global_result_case_num - 1,
            )

        elif self.configuration.solver == st7.SolverType.stQuasiStatic:
            need_previous_result_file = self.result_case_num == 1

            if need_previous_result_file:
                return self._replace(
                    result_case_num=self.configuration.qsa_steps_per_file,
                    result_file_index=self.result_file_index - 1,
                    global_result_case_num=self.global_result_case_num - 1,
                )

            else:
                return self._replace(
                    result_case_num=self.result_case_num - 1,
                    global_result_case_num=self.global_result_case_num - 1,
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
    previous_result_frame = result_frame.get_previous_result_frame()
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
    )


class RunParams(typing.NamedTuple):
    actuator: Actuator
    stress_end: float
    scaling: Scaling
    averaging: Averaging
    relaxation: Relaxation
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
    ratchet = Ratchet(prestrain_table, scaling=run_params.scaling, averaging=run_params.averaging, relaxation=run_params.relaxation)

    # Allow a maximum of 10% of the elements to yield in a given step.
    n_steps_minor_max = math.inf
    print(f"Limiting to {n_steps_minor_max} steps per load increment - only {run_params.elem_ratio_per_iter:%} can yield.")

    working_dir = directories.get_unique_sub_dir(config.active_config.fn_working_image_base)

    fn_st7 = working_dir / "Model.st7"
    fn_db = working_dir / "history.db" \
                          ""
    shutil.copy2(config.active_config.fn_st7_base, fn_st7)

    current_result_frame = ResultFrame(
        st7_file=fn_st7,
        configuration=config.active_config,
        result_file_index=0,
        result_case_num=1,
        global_result_case_num=1,
        load_time_table=Table([XY(0.0, 0.0), XY(config.active_config.qsa_time_step_size, 0.0)])
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

        scaling.assign_centroids(init_data.elem_centroid)
        averaging.populate_radius(init_data.node_xyz, init_data.elem_conns)

        # Dummy init values
        prestrain_update = PrestrainUpdate.zero()

        # Assume the elements are evenly sized. Factor of 2 is for x and y.
        n_updates_per_iter = round(2 * run_params.elem_ratio_per_iter * len(model.entity_numbers(st7.Entity.tyPLATE)))

        set_max_iters(model, config.active_config.max_iters, use_major=True)
        model.St7RunSolver(current_result_frame.configuration.solver, st7.SolverMode.smBackgroundRun, True)

        previous_load_factor = 0.0

        old_prestrain_values = ElemVectorDict()
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
            apply_prestrain(model, prestrain_load_case_num, old_prestrain_values)

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
                    minor_acuator_input_current_raw = get_results(run_params.actuator, results, current_result_frame.result_case_num)
                    minor_acuator_input_current = update_to_include_prestrains(run_params.actuator, minor_acuator_input_current_raw, old_prestrain_values)
                    write_out_screenshot(model_window, current_result_frame)
                    write_out_to_db(db, init_data, step_num_major, step_num_minor, results, current_result_frame, prestrain_update)

                prestrain_update = incremental_element_update_list(
                    previous_prestrain_update=prestrain_update,
                    num_allowed=n_updates_per_iter,
                    existing_prestrain_priority_factor=run_params.existing_prestrain_priority_factor,
                    elem_volume=init_data.elem_volume,
                    ratchet=ratchet,
                    minor_prev_prestrain=old_prestrain_values,
                    minor_acuator_input_current=minor_acuator_input_current,
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
                    if DEBUG_CASE_FOR_EVERY_INCREMENT:
                        prestrain_load_case_num = create_load_case(model, step_name())

                    # Add the next step
                    current_result_frame = add_increment(model, current_result_frame, this_load_factor, step_name(), advance_result_case=not DEBUG_CASE_FOR_EVERY_INCREMENT)
                    set_load_increment_table(model, current_result_frame, this_load_factor, prestrain_load_case_num)

                    # Make sure we're starting from the last case
                    set_restart_for(model, current_result_frame)

                    apply_prestrain(model, prestrain_load_case_num, prestrain_update.elem_prestrains)

                    # Keep track of the old results...
                    step_num_minor = next(minor_step_iter)
                    old_prestrain_values = prestrain_update.elem_prestrains

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
            for elem, idx, val in prestrain_update.elem_prestrains.as_single_values():
                scale_key = (elem, idx)
                ratchet.update_minimum(True, scale_key, val)

            previous_load_factor = this_load_factor

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

    #relaxation = LimitedIncreaseRelaxation(0.01)
    relaxation = PropRelax(0.1)
    scaling = SpacedStepScaling(y_depth=0.25, spacing=0.6, amplitude=0.2, hole_width=0.051)
    #scaling = CosineScaling(y_depth=0.25, spacing=0.4, amplitude=0.2)
    averaging = AveInRadius(0.25)
    #averaging = NoAve()

    run_params = RunParams(
        actuator=Actuator.e_local,
        stress_end=401.0,
        scaling=scaling,
        averaging=averaging,
        relaxation=relaxation,
        dilation_ratio=0.008,  # 0.8% expansion, according to Jerome
        n_steps_major=20,
        elem_ratio_per_iter=0.0001,
        existing_prestrain_priority_factor=5.0,
    )

    main(run_params)


# Combine to one video with "C:\Utilities\ffmpeg-20181212-32601fb-win64-static\bin\ffmpeg.exe -f image2 -r 12 -i Case-%04d.png -vcodec libx264 -profile:v high444 -refs 16 -crf 0 out.mp4"

