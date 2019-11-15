import itertools
import random
import typing
import enum
import collections

import st7
import pathlib
import bisect
import math
import shutil

from averaging import Averaging, NoAve, AveInRadius
from common_types import T_Elem, XY, XYZ, ElemVectorDict
from relaxation import Relaxation, PropRelax
from scaling import Scaling, SpacedStepScaling, CosineScaling
import directories

# To make reproducible
random.seed(123)

RATCHET_AT_INCREMENTS = True
DEBUG_CASE_FOR_EVERY_INCREMENT = True

fn_st7_base = pathlib.Path(r"E:\Simulations\CeramicBands\v3\Test 9C-Contact-SD2.st7")
fn_working_image_base = pathlib.Path(r"E:\Simulations\CeramicBands\v3-pics")

SCREENSHOT_RES = st7.CanvasSize(1920, 1200)

class Actuator(enum.Enum):
    """The value used to ratchet up the prestrain."""
    S11 = enum.auto()
    SvM = enum.auto()
    XX = enum.auto()
    local = enum.auto()

    def nice_name(self) -> str:
        if self == Actuator.S11:
            return "Principal 11 Stress"

        elif self == Actuator.SvM:
            return "vM Stress"

        elif self == Actuator.XX:
            return "XX Global"

        elif self == Actuator.XX:
            return "Local Directional"

        else:
            raise ValueError(self)


class Table:
    data: typing.Tuple[XY] = None
    _data_x: typing.Tuple[float] = None
    max_abs_y: float

    """Keeps track of some xy data"""

    def __init__(self, data: typing.Sequence[XY]):

        # Fail if it's not sorted, or if any compare equal.
        if not all(data[i].x < data[i+1].x for i in range(len(data)-1)):
            raise ValueError("Expecting the data to be sorted and unique.")


        self.data = tuple(data)
        self._data_x = tuple(xy.x for xy in self.data)
        self.max_abs_y = max(abs(xy.y) for xy in self.data)

    def interp(self, x: float) -> float:

        # If we're off the botton or top, just return the final value.
        if x <= self._data_x[0]:
            return self.data[0].y

        elif self._data_x[-1] <= x:
            return self.data[-1].y

        # Actually have to look up / interpolate.
        index_lower_or_equal = bisect.bisect_right(self._data_x, x) - 1

        # Off the end.
        if index_lower_or_equal == len(self.data)-1:
            return self.data[-1].y

        low = self.data[index_lower_or_equal]

        high = self.data[index_lower_or_equal + 1]

        # If we're right on the point, return it!
        if math.isclose(x, low.x):
            return low.y

        x_range = high.x - low.x
        y_range = high.y - low.y
        grad = y_range / x_range
        delta_x = x - low.x

        return low.y + delta_x * grad


    def min_val(self) -> float:
        return self.data[0].y


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


def apply_prestrain(model: st7.St7Model, case_num: int, elem_to_ratio: typing.Dict[int, XYZ]):
    """Apply all the prestrains"""

    for plate_num, prestrain_val in elem_to_ratio.items():
        prestrain = st7.Vector3(x=prestrain_val.x, y=prestrain_val.y, z=prestrain_val.z)
        model.St7SetPlatePreLoad3(plate_num, case_num, st7.PreLoadType.plPlatePreStrain, prestrain)


def setup_model_window(model_window: st7.St7ModelWindow, results: st7.St7Results, case_num: int):
    model_window.St7SetPlateResultDisplay_None()
    model_window.St7SetWindowResultCase(case_num)
    model_window.St7SetEntityContourIndex(st7.Entity.tyPLATE, st7.PlateContour.ctPlatePreStrainMagnitude)
    results.St7SetDisplacementScale(5.0, st7.ScaleType.dsAbsolute)
    model_window.St7RedrawModel(True)


def write_out_screenshot(model_window: st7.St7ModelWindow, results: st7.St7Results, case_num: int, fn: str):

    setup_model_window(model_window, results, case_num)
    #draw_area = model_window.St7GetDrawAreaSize()
    model_window.St7ExportImage(fn, st7.ImageType.itPNG, SCREENSHOT_RES.width, SCREENSHOT_RES.height)
    #model_window.St7ExportImage(fn, st7.ImageType.itPNG, EXPORT_FACT * draw_area.width, EXPORT_FACT * draw_area.height)


def get_stress_results(phase_change_actuator: Actuator, results: st7.St7Results, case_num: int) -> ElemVectorDict:
    """Get the results from the result file which will be used to re-apply the prestrain."""

    if phase_change_actuator == Actuator.S11:
        res_type = st7.PlateResultType.rtPlateStress
        res_sub_type = st7.PlateResultSubType.stPlateCombined
        index_list = [
            st7.St7API.ipPlateCombPrincipal11,
            st7.St7API.ipPlateCombPrincipal11,
            st7.St7API.ipPlateCombPrincipal11,
        ]

    elif phase_change_actuator == Actuator.SvM:
        res_type = st7.PlateResultType.rtPlateStress
        res_sub_type = st7.PlateResultSubType.stPlateCombined
        index_list = [
            st7.St7API.ipPlateCombVonMises,
            st7.St7API.ipPlateCombVonMises,
            st7.St7API.ipPlateCombVonMises,
        ]

    elif phase_change_actuator == Actuator.XX:
        res_type = st7.PlateResultType.rtPlateStress
        res_sub_type = st7.PlateResultSubType.stPlateGlobal
        index_list = [
            st7.St7API.ipPlateGlobalXX,
            st7.St7API.ipPlateGlobalXX,
            st7.St7API.ipPlateGlobalXX,
        ]

    elif phase_change_actuator == Actuator.local:
        res_type = st7.PlateResultType.rtPlateStress
        res_sub_type = st7.PlateResultSubType.stPlateLocal
        index_list = [
            st7.St7API.ipPlateLocalxx,
            st7.St7API.ipPlateLocalyy,
            st7.St7API.ipPlateLocalzz,
        ]

    else:
        raise ValueError(phase_change_actuator)

    def one_plate_result(plate_num):
        res_array = results.St7GetPlateResultArray(
            res_type,
            res_sub_type,
            plate_num,
            case_num,
            st7.SampleLocation.spCentroid,
            st7.PlateSurface.psPlateMidPlane,
            0,
        )
        if res_array.num_points != 1:
            raise ValueError()

        result_values = [res_array.results[index] for index in index_list]
        return XYZ(*result_values)

    raw_dict = {plate_num: one_plate_result(plate_num) for plate_num in results.model.entity_numbers(st7.Entity.tyPLATE)}
    return ElemVectorDict(raw_dict)

class PrestrainUpdate(typing.NamedTuple):
    elem_prestrains: ElemVectorDict
    updated_this_round: int
    not_updated_this_round: int
    prestrained_overall: int
    update_ratio: float


def incremental_element_update_list(
        num_allowed: int,
        existing_prestrain_priority_factor: float,
        elem_volume: typing.Dict[int, float],
        ratchet: Ratchet,
        minor_prev_strain: ElemVectorDict,
        minor_current_stress: ElemVectorDict,
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

    #old_prestrains = candidate_strains(False, minor_prev_stress)
    new_prestrains_all = candidate_strains(minor_current_stress)

    old_prestrains = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in minor_prev_strain.as_single_values()}
    increased_prestrains = {key: val for key, val in new_prestrains_all.items() if abs(val) > abs(old_prestrains.get(key, 0.0))}

    # Use the current stress results to choose the new elements.
    minor_current_stress = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in minor_current_stress.as_single_values()}

    def priority(elem_idx_val):
        """Takes in a prestrain item. This decides which prestrains get updated this round."""
        elem_idx = (elem_idx_val[0], elem_idx_val[1])
        current_stress_unscaled = minor_current_stress[elem_idx]

        existing_scale_ratio = abs(old_prestrains.get(elem_idx_val, 0.0)) / ratchet.table.max_abs_y
        scale_factor = 1.0 + existing_prestrain_priority_factor * existing_scale_ratio
        current_stress_scaled = current_stress_unscaled * ratchet.scaling.get_x_scale_factor(elem_idx) * scale_factor
        return abs(current_stress_scaled)

    # Get the top N elements.
    all_new = ( (elem_idx[0], elem_idx[1], val) for elem_idx, val in increased_prestrains.items())
    all_new_sorted = sorted(all_new, key=priority, reverse=True)

    top_n_new = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in all_new_sorted[0:num_allowed]}
    new_count = len(top_n_new)
    left_over_count = min(0, len(all_new_sorted) - new_count)

    # Build the new pre-strain dictionary out of old and new values.
    old_strain_single_vals = {id_key(elem_idx_val): val(elem_idx_val) for elem_idx_val in minor_prev_strain.as_single_values()}
    combined_final_single_values = {**old_strain_single_vals, **top_n_new}
    single_vals_out = [ (elem, idx, val) for (elem, idx), val in combined_final_single_values.items()]
    total_out = sum(1 for _, _, val in single_vals_out if abs(val))

    # Update the ratchet settings for the one we did ramp up.
    for elem_idx, val in top_n_new.items():
        ratchet.update_minimum(True, elem_idx, val)

    # Work out now much additional dilation has been introduced.
    extra_dilation = [
        (elem_idx, val - old_strain_single_vals.get(elem_idx, 0.0))
        for elem_idx, val in top_n_new.items()
    ]

    extra_dilation_norm = sum(elem_volume[elem_idx[0]] * abs(val) for elem_idx, val in extra_dilation)


    return PrestrainUpdate(
        elem_prestrains=ElemVectorDict.from_single_values(True, single_vals_out),
        updated_this_round=new_count,
        not_updated_this_round=left_over_count,
        prestrained_overall=total_out,
        update_ratio=extra_dilation_norm,
    )





def make_fn(actuator: Actuator, n_steps: int, scaling, averaging, stress_start, stress_end, dilation_ratio, relaxation, elem_ratio_per_iter, existing_prestrain_priority_factor) -> pathlib.Path:
    """Makes the base Strand7 model name."""
    new_stem = fn_st7_base.stem + f" - {actuator.name} {stress_start} to {stress_end} DilRat={dilation_ratio} Steps={n_steps} {scaling} {averaging} {relaxation} ElemRatio={elem_ratio_per_iter} ExistingPriority={existing_prestrain_priority_factor}"

    new_name = new_stem + fn_st7_base.suffix
    return fn_st7_base.with_name(new_name)


def fn_append(fn_orig: pathlib.Path, extra_bit) -> pathlib.Path:
    new_stem = f"{fn_orig.stem} - {extra_bit}"
    new_name = new_stem + fn_orig.suffix
    return fn_orig.with_name(new_name)


def add_increment(model: st7.St7Model, stage, inc_name) -> int:
    model.St7AddNLAIncrement(stage, inc_name)
    this_inc = model.St7GetNumNLAIncrements(stage)
    return this_inc


def set_load_increment_table(model: st7.St7Model, stage, freedom_case, load_case_bending, this_load_factor, this_inc, new_case_num):
    # Set the load and freedom case - can use either method. Don't use both though!
    model.St7SetNLAFreedomIncrementFactor(stage, this_inc, freedom_case, this_load_factor)
    for iLoadCase in range(1, model.St7GetNumLoadCase() + 1):
        if iLoadCase == load_case_bending:
            factor = this_load_factor

        elif iLoadCase == new_case_num:
            factor = 1.0

        else:
            factor = 0.0

        model.St7SetNLALoadIncrementFactor(stage, this_inc, iLoadCase, factor)

def main(
        actuator: Actuator,
        stress_end: float,
        scaling: Scaling,
        averaging: Averaging,
        relaxation: Relaxation,
        dilation_ratio: float,
        n_steps_major: int,
        #n_steps_minor_max: int,
        elem_ratio_per_iter: float,
        existing_prestrain_priority_factor: float,
):

    LOAD_CASE_BENDING = 1
    FREEDOM_CASE = 1
    STAGE = 0

    STRESS_START = 400

    prestrain_table = Table([
        XY(0.0, 0.0),
        XY(STRESS_START, 0.0),
        XY(stress_end, -1 * dilation_ratio),
        XY(stress_end + 200, -1 * dilation_ratio),
    ])

    #relaxation_param_dimensionless = 80/n_steps * relaxation_param
    ratchet = Ratchet(prestrain_table, scaling=scaling, averaging=averaging, relaxation=relaxation)

    # Allow a maximum of 10% of the elements to yield in a given step.
    #n_steps_minor_max = int(0.2 / elem_ratio_per_iter)
    n_steps_minor_max = math.inf
    print(f"Limiting to {n_steps_minor_max} steps per load increment - only {elem_ratio_per_iter:%} can yield.")

    fn_st7 = make_fn(actuator, n_steps_major, scaling, averaging, STRESS_START, stress_end, dilation_ratio, relaxation, elem_ratio_per_iter, existing_prestrain_priority_factor)
    shutil.copy2(fn_st7_base, fn_st7)

    fn_res = fn_st7.with_suffix(".NLA")
    fn_restart = fn_st7.with_suffix(".SRF")

    # Image file names
    fn_png = fn_st7.with_suffix(".png")
    fn_png_full = fn_append(fn_png, "FullLoad")
    fn_png_unloaded = fn_append(fn_png, "Unloaded")

    if DEBUG_CASE_FOR_EVERY_INCREMENT:
        working_image_dir = directories.get_unique_sub_dir(fn_working_image_base)
        dont_make_model_window = False
        with open(working_image_dir / "Meta.txt", "w") as f_meta:
            f_meta.write(str(fn_st7))

        print(f"Saving images here: {working_image_dir}")

    else:
        working_image_dir = ""
        dont_make_model_window = True

    with st7.St7Model(fn_st7, r"C:\Temp") as model, model.St7CreateModelWindow(dont_make_model_window) as model_window:

        elem_centroid = {
            elem_num: model.St7GetElementCentroid(st7.Entity.tyPLATE, elem_num, 0)
            for elem_num in model.entity_numbers(st7.Entity.tyPLATE)
            }

        node_xyz = {node_num: model.St7GetNodeXYZ(node_num) for node_num in model.entity_numbers(st7.Entity.tyNODE)}
        elem_conns = {plate_num: model.St7GetElementConnection(st7.Entity.tyPLATE, plate_num) for plate_num in model.entity_numbers(st7.Entity.tyPLATE)}
        elem_volume = {plate_num: model.St7GetElementData(st7.Entity.tyPLATE, plate_num) for plate_num in model.entity_numbers(st7.Entity.tyPLATE)}

        scaling.assign_centroids(elem_centroid)
        averaging.populate_radius(node_xyz, elem_conns)

        print(fn_st7.stem)

        # Assume the elements are evenly sized. Factor of 2 is for x and y.
        n_updates_per_iter = round(2 * elem_ratio_per_iter * len(model.entity_numbers(st7.Entity.tyPLATE)))

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

        model.St7SetResultFileName(fn_res)
        model.St7SetStaticRestartFile(fn_restart)
        model.St7RunSolver(st7.SolverType.stNonlinearStaticSolver, st7.SolverMode.smBackgroundRun, True)

        previous_load_factor = 0.0

        old_prestrain_values = ElemVectorDict()
        for step_num_major in range(n_steps_major):

            # Get the results from the last major step.
            with model.open_results(fn_res) as results:
                last_case = results.primary_cases[-1]
                this_case_results_major = get_stress_results(actuator, results, last_case)
                write_out_screenshot(model_window, results, last_case, working_image_dir / f"Case-{last_case:04}.png")

            # Update the model with the new load
            this_load_factor = (step_num_major+1) / n_steps_major

            new_case_num = create_load_case(model, STAGE, f"Prestrain at Step {step_num_major + 1}")
            apply_prestrain(model, new_case_num, old_prestrain_values)

            # Add the NLA increment
            this_inc = add_increment(model, STAGE, f"Step {step_num_major+1}")
            set_load_increment_table(model, STAGE, FREEDOM_CASE, LOAD_CASE_BENDING, this_load_factor, this_inc, new_case_num)

            # Make sure we're starting from the last case
            model.St7SetNLAInitial(fn_res, last_case)

            relaxation.set_current_relaxation(previous_load_factor, this_load_factor)

            # Perform an iterative update in which only a certain proportion of the elements can "yield" at once.
            minor_prev_stress = this_case_results_major.copy()
            minor_step_iter = itertools.count()
            step_num_minor = next(minor_step_iter)
            new_count = math.inf
            while (new_count > 0) and (step_num_minor < n_steps_minor_max):

                model.St7SaveFile()
                model.St7RunSolver(st7.SolverType.stNonlinearStaticSolver, st7.SolverMode.smBackgroundRun, True)

                # Get the results from the last minor step.
                with model.open_results(fn_res) as results:
                    last_case = results.primary_cases[-1]
                    minor_current_stress = get_stress_results(actuator, results, last_case)
                    write_out_screenshot(model_window, results, last_case, working_image_dir / f"Case-{last_case:04}.png")

                #TEMP!
                model.St7SaveFile()

                prestrain_update = incremental_element_update_list(
                    num_allowed=n_updates_per_iter,
                    existing_prestrain_priority_factor=existing_prestrain_priority_factor,
                    elem_volume=elem_volume,
                    ratchet=ratchet,
                    minor_prev_strain=old_prestrain_values,
                    minor_current_stress=minor_current_stress,
                )
                #new_prestrain_values, new_count, total_count
                new_count = prestrain_update.updated_this_round
                update_bits = [
                    f"{step_num_major + 1}.{step_num_minor + 1}/{n_steps_major}",
                    f"Updated {new_count}",
                    f"Left {prestrain_update.not_updated_this_round}",
                    f"Total {prestrain_update.prestrained_overall}",
                    f"Norm {prestrain_update.update_ratio}",
                    #str(ratchet.status_update()),
                ]
                print("\t".join(update_bits))

                # Only continue if there is updating to do.
                if prestrain_update.updated_this_round > 0:
                    if DEBUG_CASE_FOR_EVERY_INCREMENT:
                        new_case_num = create_load_case(model, STAGE, f"Prestrain at Step {step_num_major + 1}.{step_num_minor}")

                        # Add the NLA increment
                        this_inc = add_increment(model, STAGE, f"Step {step_num_major + 1}.{step_num_minor}")
                        set_load_increment_table(model, STAGE, FREEDOM_CASE, LOAD_CASE_BENDING, this_load_factor, this_inc,
                                                 new_case_num)

                        # Make sure we're starting from the last case
                        model.St7SetNLAInitial(fn_res, last_case)

                    apply_prestrain(model, new_case_num, prestrain_update.elem_prestrains)

                    # Keep track of the old results...
                    step_num_minor = next(minor_step_iter)
                    old_prestrain_values = prestrain_update.elem_prestrains


            # Update the ratchet with the equilibrated results.
            for elem, idx, val in prestrain_update.elem_prestrains.as_single_values():
                scale_key = (elem, idx)
                ratchet.update_minimum(True, scale_key, val)

            model.St7SaveFile()

            previous_load_factor = this_load_factor

            relaxation.flush_previous_values()

        model.St7SaveFile()

        # Save the image of pre-strain results from the maximum load step.
        with model.open_results(fn_res) as results, model.St7CreateModelWindow(dont_really_make=False) as model_window:
            final_prestrain_case_num = results.primary_cases[-1]
            write_out_screenshot(model_window, results, final_prestrain_case_num, fn_png_full)


def create_load_case(model, stage, case_name):
    model.St7NewLoadCase(case_name)
    new_case_num = model.St7GetNumLoadCase()
    model.St7EnableNLALoadCase(stage, new_case_num)
    return new_case_num


if __name__ == "__main__":

    #relaxation = LimitedIncreaseRelaxation(0.01)
    relaxation = PropRelax(1.0)
    scaling = SpacedStepScaling(y_depth=0.25, spacing=0.6, amplitude=0.2, hole_width=0.051)
    #scaling = CosineScaling(y_depth=0.25, spacing=0.4, amplitude=0.2)
    averaging = AveInRadius(0.05)
    #averaging = NoAve()

    main(
        Actuator.local,
        stress_end=425.0,
        scaling=scaling,
        averaging=averaging,
        relaxation=relaxation,
        dilation_ratio=0.008,  # 0.8% expansion, according to Jerome
        n_steps_major=24,
        elem_ratio_per_iter=0.00035,
        existing_prestrain_priority_factor=1.0,
    )


