import dataclasses
import itertools
import os
import random
import statistics
import time
import typing
import enum
import collections
import datetime
import contextlib
import argparse
import pickle

import st7_wrap.exc
from PIL import Image

import pathlib
import math
import shutil
import networkx

from st7_wrap import st7
from st7_wrap import const

import config

from averaging import Averaging, AveInRadius, NoAve
from common_types import T_Elem, SingleValue, XY, ElemVectorDict, T_ResultDict, InitialSetupModelData, TEMP_ELEMS_OF_INTEREST, Actuator, SingleValueWithMissingDict, NodeMoveStep, IncrementType
from parameter_trend import ParameterTrend
import parameter_trend
from distribution import OrientationDistribution, distribute, random_angle_distribution_360deg, wraparound_from_zero
from relaxation import Relaxation, NoRelax, PropRelax, LimIncRelax
from scaling import Scaling, SingleHoleCentre, SpacedStepScaling, CosineScaling, NoScaling
from tables import Table
from throttle import Throttler, StoppingCriterion, Shape, ElemPreStrainChangeData, BaseThrottler, RelaxedIncreaseDecrease
import history
import model_inspect
import perturb

import directories
import state_tracker

# To make reproducible
random.seed(123)

DONT_MAKE_MODEL_WINDOW = False

LOAD_CASE_BENDING = 1
FREEDOM_CASE = 1
STAGE = 0

STRESS_START = 400

NUM_PLATE_RES_RETRIES = 20

# Make deterministic
random.seed(123456)

#fn_st7_base = pathlib.Path(r"E:\Simulations\CeramicBands\v3\Test 9C-Contact-SD2.st7")
#fn_working_image_base = pathlib.Path(r"E:\Simulations\CeramicBands\v3-pics")

#screenshot_res = st7.CanvasSize(1920, 1200)


class ModelFreedomCase(enum.Enum):
    """Values are the freedom case numbers in Strand7"""
    restraint = 1
    bending_pure = 2
    tension = 3
    bending_three_point = 4

    def load_scale_factor_for_constant_tension(self, scale_model_x: float, scale_model_y: float):
        if self == ModelFreedomCase.restraint:
            return 1.0

        elif self in (ModelFreedomCase.bending_pure, ModelFreedomCase.bending_three_point):
            # Both x and y change the tension on the surface, so have to compensate
            return 1.0 * scale_model_x / scale_model_y

        elif self == ModelFreedomCase.tension:
            # Changing y doesn't matter.
            return 1.0 * scale_model_x

        else:
            raise ValueError(self)

    @property
    def table_id(self) -> int:
        return 10 + self.value

    def active_during_increment(self, increment_type: IncrementType) -> bool:
        if self == ModelFreedomCase.restraint:
            return True

        elif self in (ModelFreedomCase.bending_pure, ModelFreedomCase.tension, ModelFreedomCase.bending_three_point):
            return increment_type == IncrementType.loading

        else:
            raise ValueError(self)


class RunParams(typing.NamedTuple):
    actuator: Actuator
    scaling: Scaling
    averaging: Averaging
    relaxation: Relaxation
    throttler: BaseThrottler
    perturbator: perturb.BasePerturbation
    n_steps_major: int
    n_steps_minor_max: int
    start_at_major_ratio: typing.Optional[float]
    existing_prestrain_priority_factor: typing.Optional[float]
    parameter_trend: ParameterTrend
    source_file_name: pathlib.Path
    randomise_orientation: typing.Union[bool, OrientationDistribution]  # False for all the same, True for all random, or OrientationDistribution instance for a distribution so defined
    override_poisson: typing.Optional[float]
    freedom_cases: typing.List[ModelFreedomCase]
    scale_model_x: float
    scale_model_y: float
    max_load_ratio: float
    unload_step: bool
    working_dir: pathlib.Path


    def summary_strings(self) -> typing.Iterable[str]:
        yield "RunParams:\n"
        for field_name, field_type in self.__annotations__.items():
            field_val = getattr(self, field_name)
            if field_type in (Actuator,):
                output_str = field_val.nice_name()

            elif field_type in (ParameterTrend,):
                yield from field_val.summary_strings()

            elif field_type == typing.List[ModelFreedomCase]:
                names_of_loads = [mfc.name for mfc in field_val if mfc != ModelFreedomCase.restraint]
                output_str = "+".join(names_of_loads)

            else:
                output_str = str(field_val)

            yield f"{field_name}\t{output_str}\n"

        yield "\n"

    def active_freedom_case_numbers(self, increment_type: IncrementType) -> typing.FrozenSet[int]:
        return frozenset(mfc.value for mfc in self.freedom_cases if mfc.active_during_increment(increment_type))


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

    def get_all_proposed_values(self, actuator: Actuator, elem_results: ElemVectorDict) -> typing.List[SingleValue]:
        """Ratchet up the table and return the value. This is done independently for each axis.
        If 'lock_in' is set, the ratchet is "moved up". Otherwise, it's just like peeking at what the value would have been."""

        idx_to_elem_to_unaveraged = collections.defaultdict(dict)

        if TEMP_ELEMS_OF_INTEREST:
            print("Unaveraged:")

        # for elem_id, idx, result_strain_raw in elem_results.as_single_values():
        # At this stage, don't average the eigenvector... seems a bit iffy!
        elem_to_eigen_vector = {}
        for sv in elem_results.as_single_values_for_actuation(actuator).values():

            # e_11: sv[elem=1] = 0.007563815286591485
            # e_xx: sv[elem=1] = ...?

            elem_to_eigen_vector[sv.elem] = sv.eigen_vector

            # Apply scaling
            scale_key = (sv.elem, sv.axis)
            stress_scaled = self.scaling.get_x_scale_factor(scale_key) * sv.value

            # Do the stress-to-prestrain lookup
            strain_raw = self.table.interp(stress_scaled)

            # Apply relaxation
            strain_relaxed = self._relaxation.relaxed(scale_key, strain_raw)

            # The unaveraged results take the ratchet value (so any locked in pre-strain is included in the averaging).
            strain_relaxed_ratcheted = self.update_minimum(False, scale_key, strain_relaxed)

            if TEMP_ELEMS_OF_INTEREST:
                if sv.elem in TEMP_ELEMS_OF_INTEREST and sv.axis == 0:
                    print(sv.elem, sv.value, strain_raw, strain_relaxed, strain_relaxed_ratcheted, sv.eigen_vector, sep='\t')

            # Save the unaveraged results
            idx_to_elem_to_unaveraged[sv.axis][sv.elem] = strain_relaxed_ratcheted

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
                single_vals.append( SingleValue(elem_id, idx, ratchet_value, elem_to_eigen_vector[elem_id]))

                if TEMP_ELEMS_OF_INTEREST:
                    if elem_id in TEMP_ELEMS_OF_INTEREST and idx == 0:
                        print(elem_id, prestrain_val, ratchet_value, sep='\t')

        # self.min_y_so_far is now updated - compose that back into the return results.
        # return ElemVectorDict.from_single_values(False, single_vals)
        return single_vals

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
    elem_prestrains_locked_in: typing.List[SingleValue]
    elem_prestrains_iteration_set: typing.List[SingleValue]
    updated_this_round: int
    not_updated_this_round: int
    prestrained_overall: int
    update_ratio: float
    this_update_time: datetime.timedelta
    update_completed_at_time: datetime.datetime
    overall_dilation_ratio_working_set: float  # This is the volume-scaled dilation ratio of the prestrains which are not locked in (the iteration)

    @staticmethod
    def zero() -> "PrestrainUpdate":
        return PrestrainUpdate(
            elem_prestrains_locked_in=[],
            elem_prestrains_iteration_set=[],
            updated_this_round=0,
            not_updated_this_round=0,
            prestrained_overall=0,
            update_ratio=0.0,
            this_update_time=datetime.timedelta(seconds=0),
            update_completed_at_time=datetime.datetime.now(),
            overall_dilation_ratio_working_set=0.0,
        )

    def locked_in_prestrains(self) -> "PrestrainUpdate":
        return self._replace(elem_prestrains_locked_in=self.elem_prestrains_iteration_set)


def compress_png(png_fn):
    if not DONT_MAKE_MODEL_WINDOW:
        image = Image.open(png_fn)
        image.save(png_fn, optimize=True, quality=95)


def apply_prestrain(model: st7.St7Model, case_num: int, elem_prestrains: typing.List[SingleValue]):
    """Apply all the prestrains"""

    elem_to_ratio = ElemVectorDict.from_single_values(True, elem_prestrains)

    for plate_num, prestrain_val in elem_to_ratio.items():
        prestrain = st7.Vector3(x=prestrain_val.xx, y=prestrain_val.yy, z=prestrain_val.zz)
        model.St7SetPlatePreLoad3(plate_num, case_num, const.PreLoadType.plPlatePreStrain, prestrain)


def setup_model_window(run_params: RunParams, model_window: st7.St7ModelWindow, case_num: int):
    # model_window.St7SetPlateResultDisplay_None()

    
    model_window.St7SetWindowResultCase(case_num)
    model_window.St7SetEntityContourIndex(const.Entity.tyPLATE, const.PlateContour.ctPlatePreStrainMagnitude)
    model_window.St7SetDisplacementScale(5.0, const.ScaleType.dsAbsolute)

    # Set the contour limits
    contour_settings_limit = model_window.St7GetEntityContourSettingsLimits(const.Entity.tyPLATE)

    prestrain_contour_max = run_params.parameter_trend.dilation_ratio.get_max_value_returned() * run_params.actuator.contour_limit_scale_factor
    new_contour_limits = dataclasses.replace(contour_settings_limit, 
        ipContourLimit=st7.St7API.clUserRange,
        ipSetMinLimit=True, 
        ipSetMaxLimit=True, 
        ipMinLimit=0.0, 
        ipMaxLimit=prestrain_contour_max
        )

    model_window.St7SetEntityContourSettingsLimits(const.Entity.tyPLATE, new_contour_limits)

    # contour_settings_style = model_window.St7GetEntityContourSettingsStyle(st7.Entity.tyPLATE)
    
    # TODO -up to here
    model_window.St7RedrawModel(True)


def write_out_screenshot(run_params: RunParams, model_window: st7.St7ModelWindow, current_result_frame: "ResultFrame"):
    setup_model_window(run_params, model_window, current_result_frame.result_case_num)

    exported_image = False
    next_wait = 0.001
    e = Warning("Logical Error")
    while not exported_image and (next_wait < 1.0):
        try:
            model_window.St7ExportImage(current_result_frame.image_file, const.ImageType.itPNG, config.active_config.screenshot_res.width, config.active_config.screenshot_res.height)
            exported_image = True

        except OSError as e:
            print(e)
            time.sleep(next_wait)
            next_wait = next_wait * 1.5

    if not exported_image:
        raise e

    compress_png(current_result_frame.image_file)


def _make_column_stats(db_case_num: int, col_x: float, elem_nums: typing.FrozenSet[int], yielded_elems: typing.Set[int], result_strain: ElemVectorDict) -> typing.Iterable[history.ColumnResult]:

    # Bin the result in an elemnt column by axis and yieled/not-yielded
    def make_key(elem_num, idx):
        return (elem_num in yielded_elems, history.ContourKey.from_idx_total_strain(idx))

    segments = collections.defaultdict(list)

    SKIP_ZZ_STRAIN = True
    if SKIP_ZZ_STRAIN:
        allowable_idx = {0, 1}
    else:
        allowable_idx = {0, 1, 2}

    for elem_num in elem_nums:
        for sv in result_strain.get_one_elem_single_values(elem_num):
            if sv.axis in allowable_idx:
                key = make_key(elem_num, sv.axis)
                segments[key].append(sv.value)

    for (yielded, contour_key), res_list in segments.items():
        yield history.ColumnResult(
            result_case_num=db_case_num,
            x=col_x,
            yielded=yielded,
            contour_key=contour_key,
            minimum=min(res_list),
            mean=statistics.fmean(res_list),
            maximum=max(res_list),
        )

def _transformation_band_summary(db_case_num: int, element_nums: typing.Iterable[T_Elem], init_data: InitialSetupModelData, elem_to_sum_of_values: typing.Dict[T_Elem, float]) -> history.TransformationBand:
    elem_x_pos = [init_data.elem_centroid[elem_num].x for elem_num in element_nums]
    elem_y_pos = [init_data.elem_centroid[elem_num].y for elem_num in element_nums]
    elem_contributions = [elem_to_sum_of_values[elem_num] * init_data.elem_volume[elem_num] for elem_num in element_nums]

    width = max(elem_x_pos) - min(elem_x_pos)
    depth = max(elem_y_pos) - min(elem_y_pos)

    return history.TransformationBand(
        result_case_num=db_case_num,
        x=statistics.mean(elem_x_pos),
        elements_involved=len(element_nums),
        band_size=sum(elem_contributions),
        width=width,
        depth=depth,
    )


def write_out_to_db(
    db: history.DB, 
    init_data: InitialSetupModelData,
    current_inc: parameter_trend.CurrentInc, 
    results: st7.St7Results, 
    current_result_frame: "ResultFrame", 
    prestrain_update: PrestrainUpdate,
    result_strain: ElemVectorDict):

    # Main case data
    db_res_case = history.ResultCase(
        num=None,
        name=str(current_result_frame),
        major_inc=current_inc.major_inc,
        minor_inc=current_inc.minor_inc,
    )

    db_case_num = db.add(db_res_case)

    # Enforced displacements for F-D curves - always extract this
    def make_fd_results(node_num: int, dof: st7.DoF):
        disp = results.St7GetNodeResult(const.NodeResultType.rtNodeDisp, node_num, current_result_frame.result_case_num)
        react = results.St7GetNodeResult(const.NodeResultType.rtNodeReact, node_num, current_result_frame.result_case_num)
        return history.LoadDisplacementPoint(
            result_case_num=db_case_num,
            node_num=node_num,
            load_text=dof.rx_mz_text,
            disp_text=dof.name,
            load_val=react.results[dof.value],
            disp_val=disp.results[dof.value],
        )

    fd_curve_res = ( make_fd_results(node_num, dof) for node_num, dof in init_data.enforced_dofs[current_inc.increment_type])
    db.add_many(fd_curve_res)

    if config.active_config.record_result_history_in_db == config.HistoryToRecord.none:
        return

    # Deformed positions
    if config.active_config.record_result_history_in_db.only_deformed_node_pos:
        undef_node_xyz = {node: xyz for node, xyz in init_data.node_step_xyz.items() if node in init_data.boundary_nodes}

    else:
        undef_node_xyz = init_data

    deformed_pos = get_node_positions_deformed(undef_node_xyz, results, current_result_frame.result_case_num)
    db_node_xyz = (history.NodePosition(
        result_case_num=db_case_num,
        node_num=node_num,
        x=pos.x,
        y=pos.y,
        z=pos.z) for node_num, pos in deformed_pos.items())

    db.add_many(db_node_xyz)

    # Prestrains
    if config.active_config.record_result_history_in_db == config.HistoryToRecord.full:
        def make_prestrain_rows():
            for sv in prestrain_update.elem_prestrains_iteration_set:
                yield history.ContourValue(
                    result_case_num=db_case_num,
                    contour_key_num=history.ContourKey.from_idx_pre_strain(sv.axis).value,
                    elem_num=sv.elem,
                    value=sv.value,
                )

        db.add_many(make_prestrain_rows())

        # Result Strains
        def make_result_strains():
            for sv in result_strain.as_single_values():
                yield history.ContourValue(
                    result_case_num=db_case_num,
                    contour_key_num=history.ContourKey.from_idx_total_strain(sv.axis).value,
                    elem_num=sv.elem,
                    value=sv.value,
                )

        db.add_many(make_result_strains())

    # Column-wise statistics
    def make_column_results():
        yielded_elems = {sv.elem for sv in prestrain_update.elem_prestrains_iteration_set}
        for col_x, elem_nums in init_data.element_columns.items():
            yield from _make_column_stats(db_case_num, col_x, elem_nums, yielded_elems, result_strain)

    db.add_many(make_column_results())

    # Summaries of the transformation bands
    if config.active_config.record_result_history_in_db != config.HistoryToRecord.none:
        def make_transformation_band_summaries():
            elems_with_prestrain = {sv.elem for sv in prestrain_update.elem_prestrains_iteration_set}
            prestrained_subgraph = init_data.element_topological_graph.subgraph(elems_with_prestrain)

            # For quicker lookup...
            elem_to_sum_of_values = dict()
            for sv in prestrain_update.elem_prestrains_iteration_set:
                elem_to_sum_of_values[sv.elem] = elem_to_sum_of_values.get(sv.elem, 0.0) + sv.value

            for graph_component in networkx.connected_components(prestrained_subgraph):
                yield _transformation_band_summary(db_case_num, graph_component, init_data, elem_to_sum_of_values)

        db.add_many(sorted(make_transformation_band_summaries()))


def get_results(phase_change_actuator: Actuator, results: st7.St7Results, case_num: int) -> ElemVectorDict:
    """Get the results from the result file which will be used to re-apply the prestrain."""

    res_type = phase_change_actuator.input_result

    if phase_change_actuator == Actuator.S11:
        res_sub_type = const.PlateResultSubType.stPlateCombined
        index_list = [
            st7.St7API.ipPlateCombPrincipal11,
            st7.St7API.ipPlateCombPrincipal11,
            st7.St7API.ipPlateCombPrincipal11,
        ]

    elif phase_change_actuator == Actuator.SvM:
        res_sub_type = const.PlateResultSubType.stPlateCombined
        index_list = [
            st7.St7API.ipPlateCombVonMises,
            st7.St7API.ipPlateCombVonMises,
            st7.St7API.ipPlateCombVonMises,
        ]

    elif phase_change_actuator == Actuator.s_XX:
        res_sub_type = const.PlateResultSubType.stPlateGlobal
        index_list = [
            st7.St7API.ipPlateGlobalXX,
            st7.St7API.ipPlateGlobalXX,
            st7.St7API.ipPlateGlobalXX,
        ]

    elif phase_change_actuator in (Actuator.s_local, Actuator.e_local, Actuator.e_local_max):
        res_sub_type = const.PlateResultSubType.stPlateLocal
        index_list = [
            st7.St7API.ipPlateLocalxx,
            st7.St7API.ipPlateLocalyy,
            st7.St7API.ipPlateLocalzz,
        ]

    elif phase_change_actuator == Actuator.e_11:
        res_sub_type = const.PlateResultSubType.stPlateLocal
        index_list = [
            st7.St7API.ipPlateLocalxx,
            st7.St7API.ipPlateLocalyy,
            st7.St7API.ipPlateLocalzz,
            st7.St7API.ipPlateLocalxy,
            st7.St7API.ipPlateLocalyz,
            st7.St7API.ipPlateLocalzx,
        ]

    elif phase_change_actuator == Actuator.e_xx_only:
        res_sub_type = const.PlateResultSubType.stPlateLocal
        index_list = [
            st7.St7API.ipPlateLocalxx,
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
                    const.SampleLocation.spCentroid,
                    const.PlateSurface.psPlateMidPlane,
                    0,
                )
                worked = True

            except st7_wrap.exc.St7BaseException as e:
                if num_tries > 0:
                    time.sleep(0.001 * num_tries**2)

                num_tries += 1
                print(f"Failed with {e.__name__}, try {num_tries}/{NUM_PLATE_RES_RETRIES}. {plate_num=}, {case_num=}, {res_type=}, {res_sub_type=}")

                if num_tries == NUM_PLATE_RES_RETRIES:
                    raise e

        if res_array.num_points != 1:
            raise ValueError()

        result_values = [res_array.results[index] for index in index_list]

        if phase_change_actuator == Actuator.e_local_max:
            max_in_plane = max(result_values[0:2])
            result_values = [max_in_plane, max_in_plane, result_values[2]]

        while len(result_values) < 6:
            result_values.append(0.0)

        if phase_change_actuator.input_result == const.PlateResultType.rtPlateTotalStrain:
            return st7.StrainTensor(*result_values)

        else:
            raise ValueError("Only doing strains now!")

    raw_dict = {plate_num: one_plate_result(plate_num) for plate_num in results.model.entity_numbers(const.Entity.tyPLATE)}
    return ElemVectorDict(raw_dict)


def get_node_positions_deformed(orig_positions: T_ResultDict, results: st7.St7Results, case_num: int) -> T_ResultDict:
    """Deformed node positions - only consider those in orig_positions"""

    def one_node_pos(node_num: int, orig_pos: st7.Vector3):
        node_res = results.St7GetNodeResult(const.NodeResultType.rtNodeDisp, node_num, case_num).results
        deformation = st7.Vector3(x=node_res[0], y=node_res[1], z=node_res[2])
        return orig_pos + deformation

    return {node_num: one_node_pos(node_num, orig_pos) for node_num, orig_pos in orig_positions.items()}


def incremental_element_update_list(
        init_data: InitialSetupModelData,
        run_params: RunParams,
        ratchet: Ratchet,
        previous_prestrain_update: PrestrainUpdate,
        result_strain: ElemVectorDict,
) -> PrestrainUpdate:
    """Gets the subset of elements which should be "yielded", based on the stress."""

    def candidate_strains(res_dict):
        ratchet.scaling.assign_working_results(previous_prestrain_update.elem_prestrains_iteration_set)

        return ratchet.get_all_proposed_values(run_params.actuator, elem_results=res_dict)

    # Use the current stress or strain results to choose the new elements.
    minor_acuator_input_current_flat = result_strain.as_single_values_for_actuation(run_params.actuator)

    new_prestrains_all = candidate_strains(result_strain)

    old_prestrains = SingleValueWithMissingDict((sv.id_key, sv) for sv in previous_prestrain_update.elem_prestrains_iteration_set)

    proposed_prestrains_changes_all = [
        ElemPreStrainChangeData(
            elem_num=sv.id_key[0],
            axis=sv.id_key[1],
            proposed_prestrain_val=sv.value,
            old_prestrain_val=old_prestrains[sv.id_key].value,
            result_strain_val=minor_acuator_input_current_flat[sv.id_key].value * ratchet.scaling.get_x_scale_factor(sv.id_key),
            eigen_vector_proposed=sv.eigen_vector,
            eigen_vector_old=old_prestrains[sv.id_key].eigen_vector,
            elem_volume_ratio=init_data.elem_volume_ratio[sv.id_key[0]],
        )
        for sv in new_prestrains_all
    ]

    proposed_prestrains_changes = [espcd for espcd in proposed_prestrains_changes_all if abs(espcd.proposed_change()) > config.active_config.converged_delta_prestrain]

    if TEMP_ELEMS_OF_INTEREST:
        old_TEMP = {elem_idx: val for elem_idx, val in old_prestrains.items() if elem_idx[0] in TEMP_ELEMS_OF_INTEREST}
        res_TEMP = {elem_idx: val for elem_idx, val in minor_acuator_input_current_flat.items() if elem_idx[0] in TEMP_ELEMS_OF_INTEREST}
        new_TEMP = {sv.id_key: sv for sv in new_prestrains_all if sv.elem in TEMP_ELEMS_OF_INTEREST}
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
            val_bits = [maybe_sv.value if isinstance(maybe_sv, SingleValue) else maybe_sv for maybe_sv in bits]
            all_bits = [k[0]] + val_bits
            print(*all_bits, sep='\t')

        for k in all_keys:
            print_line(k)

        print("increased_prestrains_TEMP:")
        for inc in increased_prestrains_TEMP:
            print(inc)

        print()

    # Get the working set of prestrain updates.
    # TODO - maybe add the principal stuff in here too?
    proposed_prestrains_subset = ratchet.throttler.throttle(
        init_data,
        previous_prestrain_update,
        run_params,
        proposed_prestrains_changes,
    )

    top_n_new = {sv.id_key: sv for sv in (epscd.to_single_value() for epscd in proposed_prestrains_subset)}

    new_count = len(top_n_new)
    left_over_count = max(0, len(proposed_prestrains_changes) - new_count)

    # Build the new pre-strain dictionary out of old and new values.
    combined_final_single_values = list({**old_prestrains, **top_n_new}.values())
    total_out = sum(1 for sv in combined_final_single_values if abs(sv.value))

    # Work out now much additional dilation has been introduced.
    extra_dilation = [
        (elem_idx, sv.value - old_prestrains[elem_idx].value)
        for elem_idx, sv in top_n_new.items()
    ]

    extra_dilation_norm = sum(init_data.elem_volume[elem_idx[0]] * abs(val) for elem_idx, val in extra_dilation)

    update_completed_at_time = datetime.datetime.now()
    this_update_time = update_completed_at_time - previous_prestrain_update.update_completed_at_time

    return PrestrainUpdate(
        elem_prestrains_locked_in=previous_prestrain_update.elem_prestrains_locked_in,
        elem_prestrains_iteration_set=combined_final_single_values,
        updated_this_round=new_count,
        not_updated_this_round=left_over_count,
        prestrained_overall=total_out,
        update_ratio=extra_dilation_norm,
        this_update_time=this_update_time,
        update_completed_at_time=update_completed_at_time,
        overall_dilation_ratio_working_set=sum(epscd.vol_scaled_prestrain_contrib() for epscd in proposed_prestrains_subset)
    )


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


    def get_next_result_frame(self, bending_load_factor: float) -> "ResultFrame":

        proposed_new_result_case_num = self.result_case_num + 1

        this_case_with_no_history = self._replace(prev_result_case=None)

        if self.configuration.solver == const.SolverType.stNonlinearStatic:
            return self._replace(
                result_case_num=proposed_new_result_case_num,
                global_result_case_num=self.global_result_case_num + 1,
                prev_result_case=this_case_with_no_history,
            )

        elif self.configuration.solver == const.SolverType.stQuasiStatic:
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
        if self.configuration.solver == const.SolverType.stNonlinearStatic:
            return self.st7_file.with_suffix(".NLA")

        elif self.configuration.solver == const.SolverType.stQuasiStatic:
            number_suffix = f"{self.result_file_index:04}"
            new_name = f"{self.st7_file.stem}_{number_suffix}.QSA"
            return self.st7_file.with_name(new_name)

        else:
            raise ValueError(self.configuration.solver)

    @property
    def restart_file(self) -> pathlib.Path:
        if self.configuration.solver == const.SolverType.stNonlinearStatic:
            return self.result_file.with_suffix(".SRF")

        elif self.configuration.solver == const.SolverType.stQuasiStatic:
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


def add_increment(model: st7.St7Model, result_frame: ResultFrame, this_load_factor, inc_name) -> ResultFrame:
    next_result_frame = result_frame.get_next_result_frame(this_load_factor)

    if result_frame.configuration.solver == const.SolverType.stNonlinearStatic:
        model.St7AddNLAIncrement(STAGE, inc_name)
        this_inc = model.St7GetNumNLAIncrements(STAGE)
        if next_result_frame.global_result_case_num != this_inc:
            raise ValueError(f"Got {this_inc} from St7GetNumNLAIncrements but {next_result_frame.global_result_case_num} from the ResultFrame...")

    elif result_frame.configuration.solver == const.SolverType.stQuasiStatic:
        # For QSA, we roll over the result files. So sometimes this will have changed.
        model.St7SetResultFileName(next_result_frame.result_file)
        model.St7SetStaticRestartFile(next_result_frame.restart_file)

    else:
        raise ValueError("sovler type")

    set_restart_for(model, next_result_frame)

    return next_result_frame


def set_restart_for(model: st7.St7Model, result_frame: ResultFrame):
    previous_result_frame = result_frame.prev_result_case
    if result_frame.configuration.solver == const.SolverType.stNonlinearStatic:
        model.St7SetNLAInitial(previous_result_frame.result_file, previous_result_frame.result_case_num)

    elif result_frame.configuration.solver == const.SolverType.stQuasiStatic:
        model.St7SetQSAInitial(previous_result_frame.result_file, previous_result_frame.result_case_num)

    else:
        raise ValueError(result_frame.configuration.solver)


def set_load_increment_table(run_params: RunParams, model: st7.St7Model, result_frame: ResultFrame, this_bending_load_factor, prestrain_load_case_num):
    # Set the load and freedom case - can use either method. Don't use both though!

    increment_type = run_params.parameter_trend.current_inc.increment_type

    if result_frame.configuration.solver == const.SolverType.stNonlinearStatic:

        for iFreedomCase in model.freedom_case_numbers():
            model_freedom_case = ModelFreedomCase(iFreedomCase)
            load_scale_factor = model_freedom_case.load_scale_factor_for_constant_tension(run_params.scale_model_x, run_params.scale_model_y)

            if model_freedom_case in run_params.freedom_cases:
                fc_factor = this_bending_load_factor * load_scale_factor

            else:
                fc_factor = 0.0

            model.St7SetNLAFreedomIncrementFactor(STAGE, result_frame.result_case_num, iFreedomCase, fc_factor)

        for iLoadCase in model.load_case_numbers():
            if iLoadCase == LOAD_CASE_BENDING:
                factor = this_bending_load_factor

            elif iLoadCase == prestrain_load_case_num:
                factor = 1.0

            else:
                factor = 0.0

            model.St7SetNLALoadIncrementFactor(STAGE, result_frame.result_case_num, iLoadCase, factor)

    elif result_frame.configuration.solver == const.SolverType.stQuasiStatic:
        # Enable / disable the load and freedom cases.
        for iFreedomCase in model.freedom_case_numbers():
            if iFreedomCase in run_params.active_freedom_case_numbers(increment_type):
                model.St7EnableTransientFreedomCase(iFreedomCase)

            else:
                model.St7DisableTransientFreedomCase(iFreedomCase)

        for iLoadCase in model.load_case_numbers():
            should_be_enabled = iLoadCase in (LOAD_CASE_BENDING, prestrain_load_case_num)
            if should_be_enabled:
                model.St7EnableTransientLoadCase(iLoadCase)

            else:
                model.St7DisableTransientLoadCase(iLoadCase)

        # Update the tables so we get the right bending_pure factor.
        for model_freedom_case in run_params.freedom_cases:
            load_scale_factor = model_freedom_case.load_scale_factor_for_constant_tension(run_params.scale_model_x, run_params.scale_model_y)
            scaled_table = result_frame.load_time_table.copy_scaled(1.0, load_scale_factor)

            model.St7SetTableTypeData(
                const.TableType.ttVsTime,
                model_freedom_case.table_id,
                len(scaled_table.data),
                scaled_table.as_flat_doubles(),
            )

    else:
        raise ValueError(result_frame.configuration.solver)


def set_max_iters(model: st7.St7Model, max_iters: typing.Optional[config.MaxIters], use_major:bool):
    """Put a hard out at some number of increments."""

    if max_iters:
        iter_num = max_iters.major_step if use_major else max_iters.minor_step
        model.St7SetSolverDefaultsLogical(const.SolverDefaultLogical.spAllowExtraIterations, False)
        model.St7SetSolverDefaultsInteger(const.SolverDefaultInteger.spMaxIterationNonlin, iter_num)


def _override_poisson(model: st7.St7Model, target_rho: float):
    for prop_num in model.property_numbers(const.Property.ptPLATEPROP):
        existing_prop = model.St7GetPlateIsotropicMaterial(prop_num)
        new_prop = dataclasses.replace(existing_prop, ipPlateIsoPoisson=target_rho)
        model.St7SetPlateIsotropicMaterial(prop_num, new_prop)

    for prop_num in model.property_numbers(const.Property.ptBRICKPROP):
        raise ValueError("Time to do St7SetBrickIsotropicMaterial!")


def _initial_scale_node_coords(run_params: RunParams, model: st7.St7Model):
    # Run this before anything else!
    for node_num in model.entity_numbers(const.Entity.tyNODE):
        node_xyz = model.St7GetNodeXYZ(node_num)
        new_xyz = node_xyz._replace(x=node_xyz.x * run_params.scale_model_x, y=node_xyz.y * run_params.scale_model_y)
        model.St7SetNodeXYZ(node_num, new_xyz)


def initial_setup(run_params: RunParams, model: st7.St7Model, initial_result_frame: ResultFrame) -> InitialSetupModelData:

    node_step_xyz = dict()
    node_step_xyz[NodeMoveStep.original] = {node_num: model.St7GetNodeXYZ(node_num) for node_num in model.entity_numbers(const.Entity.tyNODE)}

    _initial_scale_node_coords(run_params, model)

    node_step_xyz[NodeMoveStep.scaled] = {node_num: model.St7GetNodeXYZ(node_num) for node_num in model.entity_numbers(const.Entity.tyNODE)}

    model.St7EnableSaveRestart()
    model.St7EnableSaveLastRestartStep()

    elem_centroid = {
        elem_num: model.St7GetElementCentroid(const.Entity.tyPLATE, elem_num, 0)
        for elem_num in model.entity_numbers(const.Entity.tyPLATE)
    }

    if run_params.override_poisson is not None:
        _override_poisson(model, run_params.override_poisson)

    # Element orientation, if needed.
    if run_params.randomise_orientation == False:
        elem_axis_angle_deg = {elem_num: 0.0 for elem_num in model.entity_numbers(const.Entity.tyPLATE)}

    elif run_params.randomise_orientation == True:
        elem_axis_angle_deg = dict()
        for elem_num in model.entity_numbers(const.Entity.tyPLATE):
            rand_ang = random.random() * 360
            elem_axis_angle_deg[elem_num] = rand_ang
            model.St7SetPlateXAngle1(elem_num, rand_ang)

    elif isinstance(run_params.randomise_orientation, OrientationDistribution):
        t1 = time.time()
        elem_axis_angle_deg = random_angle_distribution_360deg(run_params.randomise_orientation, elem_centroid)
        t2 = time.time()
        print(f"Took {t2-t1} secs to get orientation distribution")

        for elem_num, elem_angle in elem_axis_angle_deg.items():
            model.St7SetPlateXAngle1(elem_num, elem_angle)

    else:
        raise ValueError("Unhandled case")

    # Perturb the nodes (for wedge inset, etc)
    node_xyz_perturbed = run_params.perturbator.update_node_locations(node_step_xyz[NodeMoveStep.scaled])
    for iNode, xyz in node_xyz_perturbed.items():
        model.St7SetNodeXYZ(iNode, xyz)

    node_step_xyz[NodeMoveStep.perturbed] = node_xyz_perturbed

    elem_conns = {
        plate_num: model.St7GetElementConnection(const.Entity.tyPLATE, plate_num) for
        plate_num in model.entity_numbers(const.Entity.tyPLATE)
    }

    elem_volume = {
        plate_num: model.St7GetElementData(const.Entity.tyPLATE, plate_num, res_case_num=0) for
        plate_num in model.entity_numbers(const.Entity.tyPLATE)
    }

    total_volume = sum(elem_volume.values())
    elem_volume_ratio = {elem: vol/total_volume for elem, vol in elem_volume.items()}

    if initial_result_frame.configuration.solver == const.SolverType.stNonlinearStatic:
        # If there's no first increment, create one.
        starting_incs = model.St7GetNumNLAIncrements(STAGE)
        if starting_incs == 0:
            model.St7AddNLAIncrement(STAGE, "Initial")

        elif starting_incs == 1:
            pass

        else:
            raise Exception("Already had increments?")

        model.St7EnableNLALoadCase(STAGE, LOAD_CASE_BENDING)
        for iFC in model.freedom_case_numbers():
            if iFC in run_params.active_freedom_case_numbers(IncrementType.loading):
                model.St7EnableNLAFreedomCase(STAGE, iFC)

            else:
                model.St7DisableNLAFreedomCase(STAGE, iFC)

    else:
        # Make the time table a single row.
        model.St7SetNumTimeStepRows(1)
        model.St7SetTimeStepData(1, 1, 1, initial_result_frame.configuration.qsa_time_step_size)
        model.St7SetSolverDefaultsLogical(const.SolverDefaultLogical.spAppendRemainingTime, False)  # Whole table is always just one step.

        model.St7EnableTransientLoadCase(LOAD_CASE_BENDING)
        for iFC in model.freedom_case_numbers():
            if iFC in run_params.active_freedom_case_numbers(IncrementType.loading):
                model.St7EnableTransientFreedomCase(iFC)

            else:
                model.St7DisableTransientFreedomCase(iFC)

        # Set up the tables which will drive the bending_pure load.
        for model_freedom_case in run_params.freedom_cases:
            load_scale_factor = model_freedom_case.load_scale_factor_for_constant_tension(run_params.scale_model_x, run_params.scale_model_y)
            scaled_table = initial_result_frame.load_time_table.copy_scaled(1.0, load_scale_factor)
            model.St7NewTableType(
                const.TableType.ttVsTime,
                model_freedom_case.table_id,
                len(scaled_table.data),
                f"Load Factor ({model_freedom_case.name})",
                scaled_table.as_flat_doubles(),
            )

            model.St7SetTransientLoadTimeTable(LOAD_CASE_BENDING, model_freedom_case.table_id, False)

            for iFC in model.freedom_case_numbers():
                fc_table_id = model_freedom_case.table_id if iFC in run_params.active_freedom_case_numbers(IncrementType.loading) else 0
                model.St7SetTransientFreedomTimeTable(iFC, fc_table_id, False)

    # For all solvers.
    model.St7SetResultFileName(initial_result_frame.result_file)
    model.St7SetStaticRestartFile(initial_result_frame.restart_file)


    return InitialSetupModelData(
        node_step_xyz=node_step_xyz,
        elem_centroid=elem_centroid,
        elem_conns=elem_conns,
        elem_volume=elem_volume,
        elem_volume_ratio=elem_volume_ratio,
        elem_axis_angle_deg=elem_axis_angle_deg,
        boundary_nodes=model_inspect.get_boundary_nodes(model),
        element_columns=model_inspect.get_element_columns(elem_centroid, elem_volume),
        enforced_dofs=model_inspect.get_enforced_displacements(run_params, model),
        element_topological_graph=model_inspect.generate_plate_edge_connectivity_graph(model),
    )


def _update_prestrain_table(run_params: RunParams, table: Table, current_inc: parameter_trend.CurrentInc):
    stress_end = run_params.parameter_trend.stress_end(current_inc)
    dilation_ratio = run_params.parameter_trend.dilation_ratio(current_inc)

    if run_params.actuator.input_result == const.PlateResultType.rtPlateStress:
        table_data = [
            XY(0.0, 0.0),
            XY(STRESS_START, 0.0),
            XY(stress_end, -1 * dilation_ratio),
            XY(stress_end + 200, -1 * dilation_ratio),
        ]

    elif run_params.actuator.input_result == const.PlateResultType.rtPlateTotalStrain:
        youngs_mod = 220000  # Hacky way!
        table_data = [
            XY(0.0, 0.0),
            XY(STRESS_START / youngs_mod, 0.0),
            XY(stress_end / youngs_mod, -1 * dilation_ratio),
            XY((stress_end + 200) / youngs_mod, -1 * dilation_ratio),
        ]

    else:
        raise ValueError(run_params.actuator.input_result)

    table.set_table_data(table_data)

def _print_update_line(run_params: RunParams, prestrain_update: PrestrainUpdate):

    update_bits = [
        f"{run_params.parameter_trend.current_inc.step_name()}/{run_params.n_steps_major}",
        f"Updated {prestrain_update.updated_this_round}",
        f"Left {prestrain_update.not_updated_this_round}",
        f"Total {prestrain_update.prestrained_overall}",
        f"Norm {prestrain_update.update_ratio}",
        f"TimeDelta {prestrain_update.this_update_time.total_seconds():1.3f}"
        #str(ratchet.status_update()),
    ]
    
    print("\t".join(update_bits))


def _results_and_screenshots(
    get_result_strains: bool,
    run_params: RunParams,
    init_data: InitialSetupModelData,
    db: history.DB,
    current_result_frame: ResultFrame,
    model: st7.St7Model,
    model_window: st7.St7ModelWindow,
    prestrain_update,
):
    # This is in a separate function so I can retry it in the annoying case of intermittent execptions
    def do_it():
        with model.open_results(current_result_frame.result_file) as results:
            if get_result_strains:
                result_strain = get_results(run_params.actuator, results, current_result_frame.result_case_num)

            else:
                result_strain = ElemVectorDict()

            write_out_screenshot(run_params, model_window, current_result_frame)
            write_out_to_db(db, init_data, run_params.parameter_trend.current_inc, results, current_result_frame, prestrain_update, result_strain)

        return result_strain

    time_to_sleep = 0.0

    while time_to_sleep < 3.0:
        try:
            return do_it()

        except (st7_wrap.exc.ERR7_ExceededResultCase, st7_wrap.exc.ERR7_APIModuleNotLicensed) as e:
            e_to_raise = e
            print(e)
            time.sleep(time_to_sleep)
            time_to_sleep = 1.5 * (time_to_sleep + 0.001)

    raise e_to_raise


def run_solver_slow_and_steady(model: st7.St7Model, solver_type: const.SolverType):
    """Runs the solver and makes sure it actually ran. It seems the network drops out which causes the solver terminate..."""

    model.St7SetUseSolverDLL(True)
    
    time_to_sleep = 0.0

    while time_to_sleep < 240.0:
        try:
            model.St7RunSolver(solver_type, const.SolverMode.smBackgroundRun, True, raise_on_termination_error=True)
            return

        except st7_wrap.exc.St7BaseException as e:
            e_to_raise = e
            print(time_to_sleep, e)
            time.sleep(time_to_sleep)
            time_to_sleep = 1.2 * (time_to_sleep + 0.1)

            try:
                model.St7Init()

            except st7_wrap.exc.St7BaseException as init_again_e:
                print(init_again_e)


    raise e_to_raise



def checkpoint():
    pass


class CheckpointState(typing.NamedTuple):

    run_params: RunParams
    ratchet: Ratchet
    current_inc: int
    prestrain_update: PrestrainUpdate
    this_load_factor: float
    prev_load_factor: float

    @staticmethod
    def starting_state() -> "CheckpointState":
        return CheckpointState(
            run_params=None,
            ratchet=None,
            current_inc=None,
            prestrain_update=None,
            this_load_factor=None,
            prev_load_factor=None
        )

def _get_pickle_fn(working_dir: pathlib.Path) -> str:
    return working_dir / "current_state.pickle"


def save_state(state: CheckpointState):
    state_fn = _get_pickle_fn(state.run_params.working_dir)
    with open(state_fn, 'wb') as fp:
        pickle.dump(state, fp)
        

def load_state(working_dir: pathlib.Path) -> CheckpointState:
    state_fn = _get_pickle_fn(working_dir)
    with open(state_fn, 'rb') as fp:
        state = pickle.load(fp)

    return state


def main(state: CheckpointState):
    if not state.ratchet:
        state.ratchet = Ratchet(
            table=Table(),
            scaling=state.run_params.scaling,
            averaging=state.run_params.averaging,
            relaxation=state.run_params.relaxation,
            throttler=state.run_params.throttler)

    # Allow a maximum of 10% of the elements to yield in a given step.

    for summary_string in state.run_params.summary_strings():
        print(summary_string.strip())
        
    for summary_string in config.active_config.summary_strings():
        print(summary_string.strip())

    fn_st7 = state.run_params.working_dir / "Model.st7"
    fn_db = state.run_params.working_dir / "history.db"


    shutil.copy2(config.active_config.fn_st7_base / state.run_params.source_file_name, fn_st7)

    pt_plot_fn = str(state.run_params.working_dir / "A-ParameterTrend.png")
    parameter_trend.save_parameter_plot(state.run_params.parameter_trend, pt_plot_fn)

    load_time_table = Table()
    load_time_table.set_table_data([XY(0.0, 0.0), XY(config.active_config.qsa_time_step_size, 0.0)])

    current_result_frame = ResultFrame(
        st7_file=fn_st7,
        configuration=config.active_config,
        result_file_index=0,
        result_case_num=1,
        global_result_case_num=1,
        load_time_table=load_time_table,
        prev_result_case=None,
    )

    with open(state.run_params.working_dir / "Meta.txt", "w") as f_meta:
        f_meta.writelines(state.run_params.summary_strings())
        f_meta.writelines(config.active_config.summary_strings())

    print(f"Working directory: {state.run_params.working_dir}")
    print()

    with contextlib.ExitStack() as exit_stack:
        model = exit_stack.enter_context(st7.St7OpenFile(fn_st7, config.active_config.scratch_dir))
        model_window = exit_stack.enter_context(model.St7CreateModelWindow(DONT_MAKE_MODEL_WINDOW))
        db = exit_stack.enter_context(history.DB(fn_db))

        init_data = initial_setup(state.run_params, model, current_result_frame)
        model.St7SaveFile()
        db.add_element_connections(init_data.elem_conns)

        # Write out the axis angles
        with open(state.run_params.working_dir / "PlateAxisAngle.txt", "w") as f_orient:
            for elem_num, angle_deg in init_data.elem_axis_angle_deg.items():
                # Angles go between 0 and 90, then start going back towards the same thing. So 91 deg is also -89 deg.
                angle_wrap = wraparound_from_zero(90, angle_deg)
                f_orient.write(f"{elem_num}\t{angle_wrap}\n")

        state.run_params.scaling.assign_centroids(init_data)
        averaging.populate_radius(init_data.node_step_xyz[NodeMoveStep.perturbed], init_data.elem_conns)

        # Dummy init values
        prestrain_update = PrestrainUpdate.zero()

        set_max_iters(model, config.active_config.max_iters, use_major=True)
        run_solver_slow_and_steady(model, current_result_frame.configuration.solver)

        previous_load_factor = 0.0

        increment_types = state.run_params.n_steps_major * [IncrementType.loading]
        if state.run_params.unload_step:
            increment_types.append(IncrementType.unloading)

        for increment_type in increment_types:
            state.run_params.parameter_trend.current_inc.inc_major()
            state.run_params.parameter_trend.current_inc.increment_type = increment_type

            # Update the model with the new load
            if increment_type == IncrementType.loading:
                this_load_factor = state.run_params.max_load_ratio * (state.run_params.parameter_trend.current_inc.major_inc + 1) / state.run_params.n_steps_major

            elif increment_type == IncrementType.unloading:
                this_load_factor = 0.0

            else:
                raise ValueError(increment_type)

            should_skip = False
            if state.run_params.start_at_major_ratio is not None:
                if increment_type == IncrementType.loading:
                    if this_load_factor < state.run_params.start_at_major_ratio:
                        should_skip = True

            if should_skip:
                print(f"Skipping major increment {state.run_params.parameter_trend.current_inc.major_inc} as load factor {this_load_factor} is below cutoff of {state.run_params.start_at_major_ratio}")

            else:
                # Get the results from the last major step.
                _results_and_screenshots(False, state.run_params, init_data, db, current_result_frame, model, model_window, prestrain_update)

                prestrain_load_case_num = create_load_case(model, state.run_params.parameter_trend.current_inc.step_name())
                apply_prestrain(model, prestrain_load_case_num, prestrain_update.elem_prestrains_locked_in)

                # Add the increment, or overwrite it
                current_result_frame = add_increment(
                    model,
                    current_result_frame,
                    this_load_factor,
                    state.run_params.parameter_trend.current_inc.step_name()
                )
                set_load_increment_table(state.run_params, model, current_result_frame, this_load_factor, prestrain_load_case_num)
            
                relaxation.set_current_relaxation(previous_load_factor, this_load_factor)

                state.run_params.parameter_trend.current_inc.inc_minor()
                new_count = math.inf

                def should_do_another_minor_iteration(new_count, minor_inc: int) -> bool:
                    return (new_count > 0) and (minor_inc <= state.run_params.n_steps_minor_max)

                while should_do_another_minor_iteration(new_count, state.run_params.parameter_trend.current_inc.minor_inc):
                    model.St7SaveFile()
                    run_solver_slow_and_steady(model, current_result_frame.configuration.solver)

                    # For the next minor increment, unless overwritten.
                    set_max_iters(model, config.active_config.max_iters, use_major=False)

                    # Get the results from the last minor step.
                    result_strain = _results_and_screenshots(True, state.run_params, init_data, db, current_result_frame, model, model_window, prestrain_update)

                    _update_prestrain_table(state.run_params, state.ratchet.table, state.run_params.parameter_trend.current_inc)

                    prestrain_update = incremental_element_update_list(
                        init_data=init_data,
                        run_params=state.run_params,
                        ratchet=state.ratchet,
                        previous_prestrain_update=prestrain_update,
                        result_strain=result_strain,
                    )

                    new_count = prestrain_update.updated_this_round
                    _print_update_line(state.run_params, prestrain_update)

                    # Apply prestrains etc to the upcoming increment.
                    if config.active_config.case_for_every_increment:
                        prestrain_load_case_num = create_load_case(model, state.run_params.parameter_trend.current_inc.step_name())

                    # Add the next step
                    current_result_frame = add_increment(model, current_result_frame, this_load_factor, state.run_params.parameter_trend.current_inc.step_name())
                    set_load_increment_table(state.run_params, model, current_result_frame, this_load_factor, prestrain_load_case_num)

                    apply_prestrain(model, prestrain_load_case_num, prestrain_update.elem_prestrains_iteration_set)

                    if config.active_config.ratched_prestrains_during_iterations:
                        # Update the ratchet settings for the one we did ramp up.
                        for sv in prestrain_update.elem_prestrains_iteration_set:
                            state.ratchet.update_minimum(True, sv.id_key, sv.value)

                    # Keep track of the old results...
                    state.run_params.parameter_trend.current_inc.inc_minor()

                    set_max_iters(model, config.active_config.max_iters, use_major=True)

                    save_state(state)

                # Tack a final increment on the end so the result case is there as expected for the start of the following major increment.
                model.St7SaveFile()
                run_solver_slow_and_steady(model, current_result_frame.configuration.solver)

                prestrain_update = prestrain_update.locked_in_prestrains()
                previous_load_factor = this_load_factor

                # Update the ratchet for what's been locked in.
                for sv in prestrain_update.elem_prestrains_locked_in:
                    state.ratchet.update_minimum(True, sv.id_key, sv.value)

                relaxation.flush_previous_values()

        model.St7SaveFile()

        # Save the image of pre-strain results from the maximum load step.
        with model.St7CreateModelWindow(dont_really_make=False) as model_window:
            _results_and_screenshots(False, state.run_params, init_data, db, current_result_frame, model, model_window, prestrain_update)


def create_load_case(model, case_name):
    model.St7NewLoadCase(case_name)
    new_case_num = model.St7GetNumLoadCase()
    model.St7EnableNLALoadCase(STAGE, new_case_num)
    return new_case_num


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Transformation bands")
    parser.add_argument("--dilation_ratio", default=0.008, type=float)
    parser.add_argument("--actuator", default=Actuator.e_local, type=Actuator.from_string, choices=list(Actuator))
    parser.add_argument("--init_variation", default=0.0, type=float)
    parser.add_argument("--init_spacing", default=0.075, type=float)

    args = parser.parse_args()

    dilation_ratio_ref = 0.008   # 0.8% expansion, according to Jerome
    relaxation = NoRelax()
    averaging = NoAve()
    throttler = RelaxedIncreaseDecrease()

    # Throttle relaxation
    exp_0_7 = parameter_trend.ExponetialDecayFunctionMinorInc(-0.7, init_val=0.5, start_at=60)
    
    # Stress End
    const_401 = parameter_trend.Constant(401)

    # Dilation Ratio
    one = parameter_trend.Constant(1.0)
    const_dilation_ratio_008 =0.008 * one
    const_dilation_ratio_008_sqrt2 =0.008 / math.sqrt(2) * one

    pt = ParameterTrend(
        throttler_relaxation=0.05 * one,
        stress_end=const_401,
        dilation_ratio= args.dilation_ratio * one,
        adj_strain_ratio_true=0.25 * one,
        scaling_ratio=one,
        overall_iterative_prestrain_delta_limit=one,
        current_inc=parameter_trend.CurrentInc(),
    )

    FINE_ELEM_LEN = 0.003703703703704
    def elem_len_mod(x: float) -> float:
        """elem_len_mod(0.52) -> 0.5 with FINE_ELEM_LEN=0.25"""
        num_elems = round(x/FINE_ELEM_LEN)
        if num_elems == 0:
            num_elems = 1

        return num_elems * FINE_ELEM_LEN

    no_scaling = NoScaling()
    scaling = SpacedStepScaling(pt=pt, y_depth=0.02, spacing=elem_len_mod(args.init_spacing), amplitude=0.5, hole_width=elem_len_mod(0.01), max_variation=args.init_variation) # 0.011112 is three elements on Fine.

    perturbator_none = perturb.NoPerturb()
    perturbator_wedge_wide = perturb.IndentCentre(10, 0.05)
    perturbator_wedge_sharp = perturb.IndentCentre(50, 0.05)
    perturbator_sphere = perturb.SphericalIndentCenter(0.2, 0.05)

    run_params = RunParams(
        actuator=args.actuator,
        scaling=scaling,
        averaging=averaging,
        relaxation=relaxation,
        throttler=throttler,
        perturbator=perturbator_none,
        n_steps_major=100,
        n_steps_minor_max=25,  # This needs to be normalised to the element size. So a fine mesh will need more iterations to stabilise.
        start_at_major_ratio=0.32,  # 0.42  # 0.38 for TestE, 0.53 for TestF
        existing_prestrain_priority_factor=None,
        parameter_trend=pt,
        source_file_name=pathlib.Path("TestH-Med.st7"),
        randomise_orientation=False,
        override_poisson=None,
        freedom_cases=[ModelFreedomCase.restraint, ModelFreedomCase.bending_pure],
        scale_model_x=1.0,  # Changing the model dimentions also scales the load.
        scale_model_y=0.5,  # 0.3 Seems good
        max_load_ratio=1.0,
        unload_step=True,
        working_dir=directories.get_unique_sub_dir(config.active_config.fn_working_image_base),
    )

    aaa = pickle.dumps(run_params)

    main(run_params)


# Combine to one video with "C:\Utilities\ffmpeg-20181212-32601fb-win64-static\bin\ffmpeg.exe -f image2 -r 12 -i Case-%04d.png -vcodec libx264 -profile:v high444 -refs 16 -crf 0 out.mp4"
# Or to an x265 video with "C:\Utilities\ffmpeg-20181212-32601fb-win64-static\bin\ffmpeg.exe -f image2 -r 30 -i Case-%04d.png -c:v libx265 out265.mp4"
