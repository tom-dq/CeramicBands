"""Some parameters need to be adjusted as the simulation
progresses. This class deals with that."""
from common_types import XY
from tables import Table

import abc

import typing


class ParameterGetter:
    @abc.abstractmethod
    def __call__(self, major_iter: int, minor_iter: int) -> float:
        raise NotImplementedError()


class ExponetialDecayFunctionMinorInc(ParameterGetter):
    _exponent: float = None

    def __init__(self, exponent: float):

        if exponent >= 0:
            raise ValueError("Does not go to zero!")

        self._exponent = exponent

    def __call__(self, major_iter: int, minor_iter: int) -> float:
        return (minor_iter+1)**self._exponent

    def __str__(self):
        return f"(minor_iter+1) ** {self._exponent}"


class TableInterpolateMinor(ParameterGetter):
    _table: Table

    def __init__(self, xy_points: typing.Sequence[XY]):
        self._table = Table(xy_points)

    def __call__(self, major_iter: int, minor_iter: int) -> float:
        return self._table.interp(minor_iter)

    def __str__(self):
        return f"Table.interp(minor_iter), T = {self._table}"


class ParameterTrend(typing.NamedTuple):
    throttler_relaxation: ParameterGetter
    stress_end: ParameterGetter

    def summary_strings(self) -> typing.Iterable[str]:
        yield "ParameterTrend:\n"
        for field_name, field_type in self._field_types.items():
            field_val = getattr(self, field_name)
            output_str = str(field_val)

            yield f"{field_name}\t{output_str}\n"

        yield "\n"
