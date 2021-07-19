
import bisect
import collections
import typing
import itertools

import networkx

from st7_wrap import st7
from st7_wrap import const

import common_types
import history

if typing.TYPE_CHECKING:
    from main import RunParams

def get_boundary_nodes(model: st7.St7Model) -> typing.FrozenSet[int]:
    """See how many plates each node is connected to as a non-generalisable way of getting the boundary."""

    elem_count = collections.Counter()

    for plate_num in model.entity_numbers(const.Entity.tyPLATE):
        nodes = model.St7GetElementConnection(const.Entity.tyPLATE, plate_num)
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


def get_enforced_displacements(run_params: "RunParams", model: st7.St7Model) -> typing.Dict[common_types.IncrementType, typing.FrozenSet[typing.Tuple[int, st7.DoF]]]:
    """Return any degrees of freedom with non-zero enforced displacements."""

    def get_one_increment_type_enforced_disps(increment_type: common_types.IncrementType):
        non_zero_enf_disp = set()

        # Cache this since we hit it all the time.
        active_freedom_case_numbers = run_params.active_freedom_case_numbers(increment_type)
        for node_restraint in model.all_node_restraints():

            if node_restraint.fc_num in active_freedom_case_numbers:
                non_zero_restraints = {dof for dof, val in node_restraint.restraints.items() if val}

                if not node_restraint.global_xyz:
                    raise ValueError(f"Haven't made this deal with UCS restraints as yet. {node_restraint}")

                for dof in non_zero_restraints:
                    non_zero_enf_disp.add( (node_restraint.node_num, dof) )

        return frozenset(non_zero_enf_disp)

    return {increment_type: get_one_increment_type_enforced_disps(increment_type) for increment_type in common_types.IncrementType}


def generate_plate_edge_connectivity_graph(model: st7.St7Model) -> networkx.Graph:
    """Graph where the "nodes" are plate elements and the "edges" are plate edges shared with other plates"""

    edge_to_plate_nums = collections.defaultdict(set)
    for plate_num in model.entity_numbers(const.Entity.tyPLATE):
        conn = model.St7GetElementConnection(const.Entity.tyPLATE, plate_num)
        if len(conn) != 4:
            raise ValueError(conn)

        for iEdge in range(4):
            n1 = conn[iEdge]
            n2 = conn[(iEdge + 1) % 4]
            edge = tuple(sorted((n1, n2)))
            edge_to_plate_nums[edge].add(plate_num)

    g = networkx.Graph()
    for edge, plate_nums in edge_to_plate_nums.items():
        g.add_nodes_from(plate_nums)
        g.add_edges_from(itertools.combinations(plate_nums, 2))

    return g
