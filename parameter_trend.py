"""Some parameters need to be adjusted as the simulation
progresses. This class deals with that."""
import numbers

import common_types
from common_types import XY
from tables import Table

import abc

import typing

import matplotlib.pyplot as plt

from config import active_config


class CurrentInc:
    major_inc: int
    minor_inc: int
    increment_type: common_types.IncrementType

    def __init__(self, major_inc: int=0, minor_inc: int=0, increment_type=None):
        self.major_inc = major_inc
        self.minor_inc = minor_inc

        if not increment_type:
            self.increment_type = list(common_types.IncrementType)[0]

    def set_incs(self, major_inc: int, minor_inc: int):
        self.major_inc = major_inc
        self.minor_inc = minor_inc

    def set_major(self, major_inc: int):
        self.major_inc = major_inc

    def inc_major(self):
        self.major_inc += 1
        self.minor_inc = 0

    def inc_minor(self):
        self.minor_inc += 1

    def step_name(self) -> str:
        return f"{self.major_inc}.{self.minor_inc}"

    def __str__(self) -> str:
        return f"CurrentInc(major_inc={self.major_inc}, minor_inc={self.minor_inc})"

class ParameterGetter:
    _cached_max_value_ever = None
    _cached_single_value = None

    @abc.abstractmethod
    def __call__(self, current_inc: CurrentInc) -> float:
        raise NotImplementedError()

    def _get_values_minor_inc(self) -> typing.Iterable[float]:
        for i in range(1_000_000):
            fake_curr_inc = CurrentInc(major_inc=None, minor_inc=i)
            yield self(fake_curr_inc)

    def get_max_value_returned(self) -> float:
        if self._cached_max_value_ever is None:
            self._cached_max_value_ever = max(self._get_values_minor_inc())

        return self._cached_max_value_ever

    def get_single_value_returned(self) -> float:
        if self._cached_single_value is None:
            all_vals = set(self._get_values_minor_inc())
            if len(all_vals) == 1:
                self._cached_single_value = all_vals.pop()

            else:
                raise ValueError(f"Expected to get one value - got {len(all_vals)}")

        return self._cached_single_value

class Constant(ParameterGetter):
    _const: float = None

    def __init__(self, const: float):
        self._const = const

    def __call__(self, current_inc: CurrentInc) -> float:
        return self._const

    def __str__(self):
        return f"{self._const}"

    def __rmul__(self, other) -> "Constant":
        if not isinstance(other, numbers.Number):
            raise TypeError(other)

        return Constant(other * self._const)


class ExponetialDecayFunctionMinorInc(ParameterGetter):
    _exponent: float = None
    _init_val: float
    _final: float
    _delta: float
    _start_at: int

    def __init__(self, exponent: float, init_val: float = 1.0, final_val: float = 0.0, start_at: int = 0):

        if exponent >= 0:
            raise ValueError("Does not go to zero!")

        self._exponent = exponent
        self._init_val = init_val
        self._final = final_val
        self._delta = (self._init_val - self._final)
        self._start_at = start_at

    def __call__(self, current_inc: CurrentInc) -> float:
        if current_inc.minor_inc <= self._start_at:
            return self._final + self._delta

        else:
            return self._delta * (current_inc.minor_inc - self._start_at) ** self._exponent + self._final

    def __str__(self):
        return f"(minor_iter+1) ** {self._exponent}"

    def __rmul__(self, other) -> "ExponetialDecayFunctionMinorInc":
        if not isinstance(other, numbers.Number):
            raise TypeError(other)

        return ExponetialDecayFunctionMinorInc(
            exponent=self._exponent,
            init_val=other * self._init_val,
            final_val=other * self._final,
            start_at=self._start_at,
        )

class TableInterpolateMinor(ParameterGetter):
    _table: Table

    def __init__(self, xy_points: typing.Sequence[XY]):
        self._table = Table()
        self._table.set_table_data(xy_points)

    def __call__(self, current_inc: CurrentInc) -> float:
        return self._table.interp(current_inc.minor_inc)

    def __str__(self):
        return f"Table.interp(minor_iter), Table({self._table.data})"


    def __rmul__(self, other) -> "TableInterpolateMinor":
        if not isinstance(other, numbers.Number):
            raise TypeError(other)

        scaled_table = self._table.copy_scaled(x_scale=1.0, y_scale=other)

        return TableInterpolateMinor(scaled_table.data)


class LineData(typing.NamedTuple):
    key: str
    x_data: typing.Tuple[float]
    y_data: typing.Tuple[float]
    y_data_display_scaled: typing.Optional[typing.Tuple[float]]


class ParameterTrend(typing.NamedTuple):
    throttler_relaxation: ParameterGetter
    stress_end: ParameterGetter
    dilation_ratio: ParameterGetter
    adj_strain_ratio_true: ParameterGetter
    scaling_ratio: ParameterGetter  # Lets you remove the scaling over time. Does not remove the strained neighbor adjustment, which is controlled by adj_strain_ratio
    overall_iterative_prestrain_delta_limit: ParameterGetter
    current_inc: CurrentInc

    @property
    def _fixed_contents(self) -> dict:
        """The functions, not the current state."""

        GOOD = (ParameterGetter,)
        BAD = (CurrentInc,)

        out_d = {}
        for name, sub_param in self._asdict().items():
            field_type = self.__annotations__[name]

            is_good = isinstance(sub_param, GOOD)
            is_bad = isinstance(sub_param, BAD)

            if is_good and not is_bad:
                out_d[name] = sub_param

            elif not is_good and is_bad:
                pass

            else:
                raise ValueError(f"What about {name} {sub_param} {field_type}?")

        return out_d

    def summary_strings(self) -> typing.Iterable[str]:
        yield "ParameterTrend:\n"
        for field_name, field_type in self._fixed_contents.items():
            field_val = getattr(self, field_name)

            if field_type is not CurrentInc:
                output_str = str(field_val)
                yield f"  {field_name}\t{output_str}\n"

        yield "\n"

    def generate_plot_data(self, max_minor_inc: int) -> typing.List[LineData]:
        """Makes plot data as the minor increment increases."""

        all_line_data = []

        x_data = range(1, max_minor_inc+1)
        x_current_inc = [CurrentInc(0, x) for x in x_data]
        for name, func in self._fixed_contents.items():

            y_data = [func(x) for x in x_current_inc]
            ld = LineData(
                key=name,
                x_data=tuple(x_data),
                y_data=tuple(y_data),
                y_data_display_scaled=None,
            )
            all_line_data.append(ld)

        return all_line_data


def save_parameter_plot(pt: ParameterTrend, out_fn: str):

    DPI = 150  # Nominal!

    all_line_data = pt.generate_plot_data(250)

    num_vert = len(all_line_data)
    num_horiz = 1
    fig, axs = plt.subplots(num_vert, num_horiz, sharex=True, figsize=(active_config.screenshot_res.width/DPI, active_config.screenshot_res.height/DPI), dpi=DPI)

    name_to_line = {}

    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0.0)

    # Different colour for each line
    prop_cycle = iter(plt.rcParams['axes.prop_cycle'])

    # Plot the graphs
    for idx, ld in enumerate(all_line_data):
        these_props = next(prop_cycle)
        line = axs[idx].plot(ld.x_data, ld.y_data, label=ld.key, **these_props)
        name_to_line[ld.key] = line

        if idx == len(all_line_data)-1:
            axs[idx].set_xlabel("Minor Iteration Number")

        # Inset title
        axs[idx].text(.5, .7, ld.key, horizontalalignment='center', transform=axs[idx].transAxes)

    #fig.legend()

    plt.savefig(out_fn, dpi=DPI)



