import inspect
import typing
import collections
import enum

import st7

class XY(typing.NamedTuple):
    x: float
    y: float


T_Elem = typing.TypeVar("T_Elem")
T_Direction = int
T_ScaleKey = typing.Tuple[T_Elem, T_Direction]
T_FactVal = typing.Tuple[float, float]
T_Result = typing.TypeVar("T_Result")
T_ResultDict = typing.Dict[int, st7.Vector3]

T_Elem_Axis = typing.Tuple[int, int]  # E.g, (2354, 0) -> Elem 2354, x axis.


class SingleValue(typing.NamedTuple):
    elem: T_Elem
    axis: T_Direction
    value: float

    @property
    def id_key(self):
        return (self.elem, self.axis)

class ElemVectorDict(dict):
    """Handy subclass to iterate over the individual values."""

    def as_single_values(self) -> typing.Iterable[ SingleValue]:
        for elem, xyz in self.items():
            yield from ( SingleValue(elem, idx, val) for idx, val in enumerate(xyz))

    def copy(self):
        """Make copy return a "ElemVectorDict", not a dict."""
        return type(self)(self)

    @classmethod
    def from_single_values(cls, fill_zeros_for_incomplete: bool, single_vals: typing.Iterable[SingleValue]) -> "ElemVectorDict":
        elem_to_idx_to_val = collections.defaultdict(dict)
        for elem, idx, val in single_vals:
            elem_to_idx_to_val[elem][idx] = val

        def make_xyz_full(idx_to_val):
            return st7.Vector3(
                x=idx_to_val[0],
                y=idx_to_val[1],
                z=idx_to_val[2],
            )

        def make_xyz_incomplete(idx_to_val):
            return st7.Vector3(
                x=idx_to_val.get(0, 0.0),
                y=idx_to_val.get(1, 0.0),
                z=idx_to_val.get(2, 0.0),
            )

        if fill_zeros_for_incomplete:
            make_xyz = make_xyz_incomplete

        else:
            make_xyz = make_xyz_full

        return ElemVectorDict( (elem, make_xyz(idx_to_val)) for elem, idx_to_val in elem_to_idx_to_val.items())

    @classmethod
    def zeros_from_element_ids(cls, elems: typing.Iterable[T_Elem]) -> "ElemVectorDict":
        return ElemVectorDict( (elem, st7.Vector3(x=0.0, y=0.0, z=0.0) ) for elem in elems)


class InitialSetupModelData(typing.NamedTuple):
    node_xyz: typing.Dict[int, st7.Vector3]
    elem_centroid: typing.Dict[int, st7.Vector3]
    elem_conns: typing.Dict[int, typing.Tuple[int, ...]]
    elem_volume: typing.Dict[int, float]
    elem_volume_ratio: typing.Dict[int, float]


def func_repr(f) -> str:
    """Returns something like the source of the function..."""
    source = inspect.getsource(f)
    source_lines = source.strip().splitlines()
    if len(source_lines) > 2:
        print(source)
        raise ValueError("Only meant for one liners, not this!")

    return f"{source_lines[0]}  {source_lines[1].strip()}"



class Actuator(enum.Enum):
    """The value used to ratchet up the prestrain."""
    S11 = enum.auto()
    SvM = enum.auto()
    s_XX = enum.auto()
    s_local = enum.auto()
    e_local = enum.auto()
    e_xx_only = enum.auto()
    e_11 = enum.auto()

    def nice_name(self) -> str:
        if self == Actuator.S11:
            return "Principal 11 Stress"

        elif self == Actuator.SvM:
            return "vM Stress"

        elif self == Actuator.s_XX:
            return "XX Global Stress"

        elif self == Actuator.e_local:
            return "Local Directional Strain"

        elif self == Actuator.e_xx_only:
            return "Strain local xx only"

        elif self == Actuator.e_11:
            return "Principal 11 Strain"

        else:
            raise ValueError(self)

    @property
    def input_result(self) -> st7.PlateResultType:
        if self in (Actuator.S11, Actuator.SvM, Actuator.s_XX, Actuator.s_local):
            return st7.PlateResultType.rtPlateStress

        elif self in (Actuator.e_local, Actuator.e_xx_only, Actuator.e_11):
            return st7.PlateResultType.rtPlateStrain

        else:
            raise ValueError(self)



# TEMP_ELEMS_OF_INTEREST = {4001, 4002, 4003, 4004, 4201,}
# TEMP_ELEMS_OF_INTEREST = {8606, 8597, 8704,}
TEMP_ELEMS_OF_INTEREST = set()