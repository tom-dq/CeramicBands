import random
import typing
import enum
import collections
import abc


import st7
import pathlib
import bisect
import math
import shutil


# To make reproducible
from averaging import Averaging, NoAveraging
from common_types import T_Elem, T_ScaleKey, T_FactVal
from scaling import Scaling, SpacedStepScaling

random.seed(123)

fn_st7_base = pathlib.Path(r"E:\Simulations\CeramicBands\v3\Test 8SD7.st7")


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


class XY(typing.NamedTuple):
    x: float
    y: float


class XYZ(typing.NamedTuple):
    x: float
    y: float
    z: float


class Table:
    data: typing.Tuple[XY] = None
    _data_x: typing.Tuple[float] = None

    """Keeps track of some xy data"""

    def __init__(self, data: typing.Sequence[XY]):

        # Fail if it's not sorted, or if any compare equal.
        if not all(data[i].x < data[i+1].x for i in range(len(data)-1)):
            raise ValueError("Expecting the data to be sorted and unique.")


        self.data = tuple(data)
        self._data_x = tuple(xy.x for xy in self.data)


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


# Relaxation controls some "backoff" when a big jump in dilation is encountered.
class Relaxation:
    _history: typing.Dict[T_ScaleKey, T_FactVal]
    _prev_factor: float = None
    _this_factor: float = None

    def set_current_relaxation(self, prev_factor: float, this_factor: float):
        self._prev_factor = prev_factor
        self._this_factor = this_factor

    @abc.abstractmethod
    def relaxed(self, scale_key, new_value) -> float:
        raise NotImplementedError


    def _lookup_key(self, scale_key, factor):
        return (scale_key, factor)


    def _get_prev_val(self, scale_key):
        # If this is the first round, there should be no previous value, so take it to be zero. Otherwise, should be something there!
        lookup_key = self._lookup_key(scale_key, self._prev_factor)
        if self._prev_factor == 0.0:
            if lookup_key in self._history:
                raise ValueError("Why is there something in history on the first step?")

            return 0.0

        else:
            return self._history[lookup_key]

    def _set_current_val(self, scale_key, new_val_relaxed):
        # Save the value for next time.
        self._history[self._lookup_key(scale_key, self._this_factor)] = new_val_relaxed

    def flush_previous_values(self):
        """Call this after doing the relaxation for a step - it clears detritus from the history."""
        keys_to_delete = [(scale_key, factor) for (scale_key, factor) in self._history.keys() if factor == self._prev_factor]
        for key in keys_to_delete:
            del self._history[key]


class ProportionalRelaxation(Relaxation):
    """Applies some relaxation to the ramp-up of the pre-strain, for example, limit it to 50% of the raw increase."""
    _incremental_ratio: float = None

    def __init__(self, incremental_ratio: float):
        self._incremental_ratio = incremental_ratio
        self._history = dict()

    def __str__(self):
        return f"{self.__class__.__name__}(incremental_ratio={self._incremental_ratio})"

    def relaxed(self, scale_key, new_value) -> float:
        """prev_factor and new_factor are the load case / freedom case factor.
           new_value is the "full" prestrain at the current step."""

        prev_val = self._get_prev_val(scale_key)

        delta_value_full = new_value - prev_val
        relaxed_value = prev_val + self._incremental_ratio * delta_value_full

        self._set_current_val(scale_key, relaxed_value)

        return relaxed_value


class LimitedIncreaseRelaxation(Relaxation):
    """Clip the new value if it's gone up to much."""
    _strain_change_limit: float = None

    def __init__(self, strain_change_limit: float):
        self._strain_change_limit = strain_change_limit
        self._history = dict()

    def __str__(self):
        return f"{self.__class__.__name__}(strain_change_limit={self._strain_change_limit})"

    def relaxed(self, scale_key, new_value) -> float:
        """prev_factor and new_factor are the load case / freedom case factor.
           new_value is the "full" prestrain at the current step."""

        prev_val = self._get_prev_val(scale_key)

        delta_value_full = new_value - prev_val

        if abs(delta_value_full) > self._strain_change_limit:
            # Need to clip it a bit.
            if delta_value_full > 0:
                relaxed_value = prev_val + self._strain_change_limit

            elif delta_value_full < 0:
                relaxed_value = prev_val - self._strain_change_limit

            else:
                raise ValueError("Zero?")

        else:
            relaxed_value = new_value

        self._set_current_val(scale_key, relaxed_value)

        return relaxed_value


class Ratchet:
    """Ratchets up a table, keeping track of which element is up to where."""
    _table: Table = None
    _relaxation: Relaxation
    _scaling: Scaling
    _averaging: Averaging
    min_y_so_far: dict
    max_stress_ever: float

    def __init__(self, table: Table, scaling: Scaling, relaxation: Relaxation, averaging: Averaging):
        self._table = table
        self._relaxation = relaxation
        self._scaling = scaling
        self._averaging = averaging
        self.min_y_so_far = dict()
        self.max_stress_ever = -1 * math.inf

    def get_all_values(self, elem_results: typing.Dict[T_Elem, XYZ]) -> typing.Dict[T_Elem, XYZ]:
        """Ratchet up the table and return the value. This is done independently for each axis."""

        idx_to_elem_to_unaveraged = collections.defaultdict(dict)

        for elem_id, stress_vector_raw in elem_results.items():
            for idx, stress_raw in enumerate(stress_vector_raw):

                self.max_stress_ever = max(self.max_stress_ever, stress_raw)

                # Apply scaling
                scale_key = (elem_id, idx)
                stress_scaled = self._scaling.get_x_scale_factor(scale_key) * stress_raw

                # Do the stress-to-prestrain lookup
                strain_raw = self._table.interp(stress_scaled)

                # Apply relaxation
                strain_relaxed = self._relaxation.relaxed(scale_key, strain_raw)

                # Save the unaveraged results
                idx_to_elem_to_unaveraged[idx][elem_id] = strain_relaxed

        # Perform averaging on a per-index basis (so the x-prestrain is only averaged with other x-prestrain.)
        for idx, unaveraged in idx_to_elem_to_unaveraged.items():
            averaged = self._averaging.average_results(unaveraged)

            # Do the ratcheting
            for elem_id, prestrain_val in averaged.items():
                ratchet_key = (elem_id, idx)

                if ratchet_key not in self.min_y_so_far:
                    self.min_y_so_far[ratchet_key] = math.inf

                self.min_y_so_far[ratchet_key] = min(prestrain_val, self.min_y_so_far[ratchet_key])

        # self.min_y_so_far is now updated - compose that back into the return results.
        return_result = dict()
        for elem_id, stress_vector_raw in elem_results.items():
            out_indices = range(len(stress_vector_raw))
            out_working_list = [self.min_y_so_far[(elem_id, idx)] for idx in out_indices]
            return_result[elem_id] = XYZ(*out_working_list)

        return return_result

    def _get_value_single(self, scale_key, x_unscaled: float) -> float:
        """Interpolates a single value."""

        self.max_stress_ever = max(self.max_stress_ever, x_unscaled)

        # Apply the scaling.
        x_scaled = self._scaling.get_x_scale_factor(scale_key) * x_unscaled

        # Look up the table value
        y_val_unrelaxed = self._table.interp(x_scaled)

        # Apply the relaxation on the looked-up value.
        y_val = self._relaxation.relaxed(scale_key, y_val_unrelaxed)

        if scale_key not in self.min_y_so_far:
            self.min_y_so_far[scale_key] = math.inf

        # Update the maximum y value seen.
        self.min_y_so_far[scale_key] = min(y_val, self.min_y_so_far[scale_key])

        return self.min_y_so_far[scale_key]


    def status_update(self) -> str:
        """Status update string"""

        val_key = {(val, key) for key,val in self.min_y_so_far.items()}
        max_val, max_elem = min(val_key)

        non_zero = [key for key,val in self.min_y_so_far.items() if val != 0.0]

        proportion_non_zero = len(non_zero) / len(self.min_y_so_far)
        return f"Elem {max_elem}\tMaxStress {self.max_stress_ever}\tStrain {max_val}\tNonzero {proportion_non_zero:.3%}"



def apply_prestrain(model: st7.St7Model, case_num: int, elem_to_ratio: typing.Dict[int, XYZ]):
    """Apply all the prestrains"""

    for plate_num, prestrain_val in elem_to_ratio.items():
        prestrain = st7.Vector3(x=prestrain_val.x, y=prestrain_val.y, z=prestrain_val.z)
        model.St7SetPlatePreLoad3(plate_num, case_num, st7.PreLoadType.plPlatePreStrain, prestrain)


def write_out_screenshot(results: st7.St7Results, case_num: int, fn: str):
    model = results.model

    model.St7CreateModelWindow()
    model.St7SetPlateResultDisplay_None()
    model.St7SetWindowResultCase(case_num)
    model.St7SetEntityContourIndex(st7.Entity.tyPLATE, st7.PlateContour.ctPlatePreStrainMagnitude)
    results.St7SetDisplacementScale(5.0, st7.ScaleType.dsAbsolute)
    model.St7RedrawModel(True)
    draw_area = model.St7GetDrawAreaSize()
    model.St7ExportImage(fn, st7.ImageType.itPNG, 2*draw_area.width, 2*draw_area.height)
    model.St7DestroyModelWindow()


def get_stress_results(phase_change_actuator: Actuator, results: st7.St7Results, case_num: int) -> typing.Dict[int, XYZ]:
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

    return {plate_num: one_plate_result(plate_num) for plate_num in results.model.entity_numbers(st7.Entity.tyPLATE)}


def make_fn(actuator: Actuator, n_steps: int, scaling, averaging, stress_start, stress_end, dilation_ratio, relaxation) -> pathlib.Path:
    """Makes the base Strand7 model name."""
    new_stem = fn_st7_base.stem + f" - {actuator.name} {stress_start} to {stress_end} DilationRatio={dilation_ratio} Steps={n_steps} {scaling} {averaging} {relaxation}"

    new_name = new_stem + fn_st7_base.suffix
    return fn_st7_base.with_name(new_name)

def fn_append(fn_orig: pathlib.Path, extra_bit) -> pathlib.Path:
    new_stem = f"{fn_orig.stem} - {extra_bit}"
    new_name = new_stem + fn_orig.suffix
    return fn_orig.with_name(new_name)


def main(actuator: Actuator, stress_end: float, scaling: Scaling, averaging: Averaging, relaxation: Relaxation, dilation_ratio: float, n_steps: int):

    LOAD_CASE_BENDING = 1
    FREEDOM_CASE = 1
    STAGE = 0

    STRESS_START = 400

    prestrain_table = Table([
        XY(0.0, 0.0),
        XY(STRESS_START, 0.0),
        XY(stress_end, -1*dilation_ratio),
        XY(stress_end+200, -1*dilation_ratio),
    ])

    #relaxation_param_dimensionless = 80/n_steps * relaxation_param
    ratchet = Ratchet(prestrain_table, scaling=scaling, averaging=averaging, relaxation=relaxation)

    fn_st7 = make_fn(actuator, n_steps, scaling, averaging, STRESS_START, stress_end, dilation_ratio, relaxation)
    shutil.copy2(fn_st7_base, fn_st7)

    fn_res = fn_st7.with_suffix(".NLA")
    fn_restart = fn_st7.with_suffix(".SRF")

    # Image file names
    fn_png = fn_st7.with_suffix(".png")
    fn_png_full = fn_append(fn_png, "FullLoad")
    fn_png_unloaded = fn_append(fn_png, "Unloaded")

    with st7.St7Model(fn_st7, r"E:\Temp") as model:

        elem_centroid = {
            elem_num: model.St7GetElementCentroid(st7.Entity.tyPLATE, elem_num, 0)
            for elem_num in model.entity_numbers(st7.Entity.tyPLATE)
            }

        node_xyz = {node_num: model.St7GetNodeXYZ(node_num) for node_num in model.entity_numbers(st7.Entity.tyNODE)}
        elem_conns = {plate_num: model.St7GetElementConnection(st7.Entity.tyPLATE, plate_num) for plate_num in model.entity_numbers(st7.Entity.tyPLATE)}

        scaling.assign_centroids(elem_centroid)
        averaging.populate_radius(node_xyz, elem_conns)

        #print(actuator, str(scaling), str(averaging), stress_end, relaxation_param)
        print(fn_st7.stem)

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
        for step_num in range(n_steps):

            # Get the results from the last step.
            with model.open_results(fn_res) as results:
                last_case = results.primary_cases[-1]
                this_case_results = get_stress_results(actuator, results, last_case)

            # Update the model
            this_load_factor = (step_num+1) / n_steps

            relaxation.set_current_relaxation(previous_load_factor, this_load_factor)
            new_prestrain_values = ratchet.get_all_values(this_case_results) # {num: ratchet.get_value(num, one_res) for num, one_res in this_case_results.items()}
            previous_load_factor = this_load_factor

            relaxation.flush_previous_values()

            print(f"{step_num+1}/{n_steps}\t{ratchet.status_update()}")

            model.St7AddNLAIncrement(STAGE, f"Step {step_num+1}")
            this_inc = model.St7GetNumNLAIncrements(STAGE)

            # Keep track of the prestrains in new load cases (just for visuals)
            model.St7NewLoadCase(f"Prestrain at Step {step_num+1}")
            new_case_num = model.St7GetNumLoadCase()
            apply_prestrain(model, new_case_num, new_prestrain_values)
            model.St7EnableNLALoadCase(STAGE, new_case_num)

            # Set the load and freedom case - can use either method. Don't use both though!
            model.St7SetNLAFreedomIncrementFactor(STAGE, this_inc, FREEDOM_CASE, this_load_factor)
            for iLoadCase in range(1, model.St7GetNumLoadCase()+1):
                if iLoadCase == LOAD_CASE_BENDING:
                    factor = this_load_factor

                elif iLoadCase == new_case_num:
                    factor = 1.0

                else:
                    factor = 0.0

                model.St7SetNLALoadIncrementFactor(STAGE, this_inc, iLoadCase, factor)

            # Make sure we're starting from the last case
            model.St7SetNLAInitial(fn_res, last_case)

            model.St7RunSolver(st7.SolverType.stNonlinearStaticSolver, st7.SolverMode.smBackgroundRun, True)

        # Save the image of pre-strain results from the maximum load step.
        with model.open_results(fn_res) as results:
            final_prestrain_case_num = results.primary_cases[-1]
            write_out_screenshot(results, final_prestrain_case_num, fn_png_full)


        model.St7SaveFile()


if __name__ == "__main__":

    #scaling = CosineScaling(y_depth=0.5, spacing=1.0, amplitude=0.1)
    relaxation = LimitedIncreaseRelaxation(0.01)
    scaling = SpacedStepScaling(y_depth=0.5, spacing=0.6, amplitude=0.1, hole_width=0.02)
    #averaging = AverageInRadius(0.25)
    averaging = NoAveraging()
    for n_steps in [150]:
        main(
            Actuator.local,
            stress_end=410.0,
            scaling=scaling,
            averaging=averaging,
            relaxation=relaxation,
            dilation_ratio=0.015,
            n_steps=n_steps)


