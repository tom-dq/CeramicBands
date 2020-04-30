import typing
import collections
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

class ElemVectorDict(dict):
    """Handy subclass to iterate over the individual values."""

    def as_single_values(self) -> typing.Iterable[ typing.Tuple[T_Elem, int, float]]:
        for elem, xyz in self.items():
            yield from ( (elem, idx, val) for idx, val in enumerate(xyz))

    def copy(self):
        """Make copy return a "ElemVectorDict", not a dict."""
        return type(self)(self)

    @classmethod
    def from_single_values(cls, fill_zeros_for_incomplete: bool, single_vals: typing.Iterable[ typing.Tuple[T_Elem, int, float]]) -> "ElemVectorDict":
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