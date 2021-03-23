import inspect
import math
import typing
import collections
import enum

import numpy.linalg

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
            return st7.PlateResultType.rtPlateTotalStrain

        else:
            raise ValueError(self)

    @property
    def needs_principal_rotation(self) -> bool:
        does_require = (
            Actuator.e_11,
        )

        if self in does_require:
            return True

        does_not_require = (
            Actuator.s_XX,
            Actuator.s_local,
            Actuator.s_local,
            Actuator.e_local,
            Actuator.e_xx_only,
        )

        if self in does_not_require:
            return False

        raise ValueError(self)


class SingleValue(typing.NamedTuple):
    elem: T_Elem
    axis: T_Direction
    value: float
    eigen_vector: typing.Optional[numpy.array] = None  # This is just for when we need to reverse a local->principal transformation

    @property
    def id_key(self) -> T_ScaleKey:
        return (self.elem, self.axis)


class SingleValueWithMissingDict(dict):

    def __missing__(self, key: T_ScaleKey) -> SingleValue:
        # Make a fake missing value and return it. Don't actually save it in the real dictionary...
        return SingleValue(
            elem=key[0],
            axis=key[1],
            value=0,
            eigen_vector=None,
        )


class ElemVectorDict(dict):
    """Handy subclass to iterate over the individual values."""

    def as_single_values(self) -> typing.Iterable[ SingleValue]:
        for elem, tensor in self.items():
            to_return = [tensor.xx, tensor.yy, tensor.zz]
            yield from ( SingleValue(elem, idx, val) for idx, val in enumerate(to_return))

    def get_one_elem_single_values(self, elem) -> typing.Iterable[SingleValue]:
        """If elem isn't in here, effectively an empty iterator rather than KeyError."""
        tensor = self.get(elem, None)
        if tensor:
            to_return = [tensor.xx, tensor.yy, tensor.zz]
            yield from (SingleValue(elem, idx, val) for idx, val in enumerate(to_return))

    def _as_principal_values(self) -> typing.Iterable[ SingleValue ]:
        for elem, tensor in self.items():
            ONLY_RETURN_MAX_EIGENVALUE = True

            values, vectors = numpy.linalg.eig(tensor.as_np_array())
            idx_max_eig = values.argmax()

            for idx, this_val in enumerate(values):
                if idx == idx_max_eig or not ONLY_RETURN_MAX_EIGENVALUE:
                    this_vect = vectors[:, idx]

                    return_axis = 0 if ONLY_RETURN_MAX_EIGENVALUE else idx
                    yield SingleValue(
                        elem=elem,
                        axis=return_axis,
                        value=this_val,
                        eigen_vector=this_vect,
                    )


    def as_single_values_for_actuation(self, actuator: Actuator) -> SingleValueWithMissingDict:
        if actuator.needs_principal_rotation:
            raw = self._as_principal_values()

        else:
            raw = self.as_single_values()

        d = SingleValueWithMissingDict()
        for one in raw:
            d[one.id_key] = one

        return d

    def copy(self):
        """Make copy return a "ElemVectorDict", not a dict."""
        return type(self)(self)

    @classmethod
    def from_single_values(cls, fill_zeros_for_incomplete: bool, single_vals: typing.Iterable[SingleValue]) -> "ElemVectorDict":
        elem_to_idx_to_val = collections.defaultdict(dict)
        elem_to_idx_to_eig = collections.defaultdict(dict)

        single_vals = list(single_vals)
        eig_is_none = [sv.eigen_vector is None for sv in single_vals]
        if all(eig_is_none):
            do_eigens = False

        elif not any(eig_is_none):
            do_eigens = True

        else:
            raise ValueError("Got some with and some without eigenvectors...")

        for sv in single_vals:
            elem_to_idx_to_val[sv.elem][sv.axis] = sv.value
            elem_to_idx_to_eig[sv.elem][sv.axis] = sv.eigen_vector 

        def make_xyz_full(elem, idx_to_val):
            return st7.StrainTensor(
                xx=idx_to_val[0],
                yy=idx_to_val[1],
                zz=idx_to_val[2],
                xy=0.0,
                yz=0.0,
                zx=0.0,
            )

        def make_xyz_incomplete(elem, idx_to_val):
            return st7.StrainTensor(
                xx=idx_to_val.get(0, 0.0),
                yy=idx_to_val.get(1, 0.0),
                zz=idx_to_val.get(2, 0.0),
                xy=0.0,
                yz=0.0,
                zx=0.0,
            )

        def make_xyz_from_eigens(elem, idx_to_val):
            # Back-rotate the from pricipal to local using the eigenvectors
            N = len(idx_to_val)
            vects = numpy.zeros(shape=(3, N))
            vals = numpy.zeros(shape=(N,))

            eigens = elem_to_idx_to_eig[elem]
            for idx, val in idx_to_val.items():
                vals[idx] = val
                vects[:, idx] = eigens[idx]

            matrix_with_shears = vects * vals * vects.T
            return st7.StrainTensor(
                xx=float(matrix_with_shears[0, 0]),
                yy=float(matrix_with_shears[1, 1]),
                zz=float(matrix_with_shears[2, 2]),
                xy=2.0 * float(matrix_with_shears[0, 1]),
                yz=2.0 * float(matrix_with_shears[1, 2]),
                zx=2.0 * float(matrix_with_shears[0, 2]),
            )

        if do_eigens:
            make_xyz = make_xyz_from_eigens

        else:
            if fill_zeros_for_incomplete:
                make_xyz = make_xyz_incomplete

            else:
                make_xyz = make_xyz_full

        return ElemVectorDict( (elem, make_xyz(elem, idx_to_val)) for elem, idx_to_val in elem_to_idx_to_val.items())

    @classmethod
    def zeros_from_element_ids(cls, elems: typing.Iterable[T_Elem]) -> "ElemVectorDict":
        return ElemVectorDict( (elem, st7.Vector3(x=0.0, y=0.0, z=0.0) ) for elem in elems)



class InitialSetupModelData(typing.NamedTuple):
    node_xyz: typing.Dict[int, st7.Vector3]
    elem_centroid: typing.Dict[int, st7.Vector3]
    elem_conns: typing.Dict[int, typing.Tuple[int, ...]]
    elem_volume: typing.Dict[int, float]
    elem_volume_ratio: typing.Dict[int, float]
    elem_axis_angle_deg: typing.Dict[int, float]
    boundary_nodes: typing.FrozenSet[int]
    element_columns: typing.Dict[float, typing.FrozenSet]


def func_repr(f) -> str:
    """Returns something like the source of the function..."""
    source = inspect.getsource(f)
    source_lines = source.strip().splitlines()
    if len(source_lines) > 2:
        print(source)
        raise ValueError("Only meant for one liners, not this!")

    return f"{source_lines[0]}  {source_lines[1].strip()}"



def test_local_princ():
    tensor_local = st7.StrainTensor(
        xx=2.360200000000e-1,
        yy=-7.758000000000e-2,
        zz=-1.899000000000e-1,
        xy=9.079330251953e-3,
        yz=-2.250140864305e-2,
        zx=-3.692339337587e-2,
    )

    MAX_11_STRAIN = 2.368989943739e-1

    assert math.isclose(abs(tensor_local), MAX_11_STRAIN)



# TEMP_ELEMS_OF_INTEREST = {4001, 4002, 4003, 4004, 4201,}
# TEMP_ELEMS_OF_INTEREST = {8606, 8597, 8704,}
# TEMP_ELEMS_OF_INTEREST = {700,}
TEMP_ELEMS_OF_INTEREST = set()