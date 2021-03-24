
import bisect
import collections
import typing
import itertools

import st7

if typing.TYPE_CHECKING:
    from main import RunParams

def get_boundary_nodes(model: st7.St7Model) -> typing.FrozenSet[int]:
    """See how many plates each node is connected to as a non-generalisable way of getting the boundary."""

    elem_count = collections.Counter()

    for plate_num in model.entity_numbers(st7.Entity.tyPLATE):
        nodes = model.St7GetElementConnection(st7.Entity.tyPLATE, plate_num)
        for node in nodes:
            elem_count[node] += 1

    # One is corner and two is an edge.
    return frozenset(iNode for iNode, plate_count in elem_count.items() if plate_count < 3)


def get_element_columns(elem_centroid: typing.Dict[int, st7.Vector3], elem_volume: typing.Dict[int, float]
                        ) -> typing.Dict[float, typing.FrozenSet[int]]:

    columns = collections.defaultdict(set)

    elem_volume_cutoff = 1.25 * min(elem_volume.values())
    small_elems = {elem for elem, vol in elem_volume.items() if vol <= elem_volume_cutoff}
    for elem in small_elems:
        x_val = round(elem_centroid[elem].x, 12)
        columns[x_val].add(elem)

    return {x_val: frozenset(elem_nums) for x_val, elem_nums in columns.items()}


def get_enforced_displacements(run_params: "RunParams", model: st7.St7Model) -> typing.FrozenSet[typing.Tuple[int, st7.DoF]]:
    """Return any degrees of freedom with non-zero enforced displacements."""

    non_zero_enf_disp = set()

    # Cache this since we hit it all the time.
    active_freedom_case_numbers = run_params.active_freedom_case_numbers
    for node_restraint in model.all_node_restraints():

        if node_restraint.fc_num in active_freedom_case_numbers:
            non_zero_restraints = {dof for dof, val in node_restraint.restraints.items() if val}

            if not node_restraint.global_xyz:
                raise ValueError(f"Haven't made this deal with UCS restraints as yet. {node_restraint}")

            for dof in non_zero_restraints:
                non_zero_enf_disp.add( (node_restraint.node_num, dof) )

    return frozenset(non_zero_enf_disp)

