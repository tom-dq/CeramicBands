import collections
import math

import pyvista
import numpy
import typing
import holoviews

import history


class ContourView(typing.NamedTuple):
    db_case_num: int
    contour_key: history.ContourKey


def make_quadmesh(
        db: history.DB,
        contour_view: ContourView,
) -> holoviews.Overlay:
    """Make a single Quadmesh representation"""

    # TODO - up to here, after adding some stuff to history.DB


    node_skeleton = history.NodePosition._all_nones()._replace(result_case_num=contour_view.db_case_num)
    node_coords = list(db.get_all_matching(node_skeleton))

    # Make the undeformed meshgrid.
    x_vals = design.gen_ordinates(stent_params.divs.Th, 0.0, stent_params.theta_arc_initial)
    y_vals = design.gen_ordinates(stent_params.divs.Z, 0.0, stent_params.length)

    # Get the bounds which are non-NaN
    idx_z = [idx.Z for idx in elem_idx_to_value.keys()]
    idx_Th = [idx.Th for idx in elem_idx_to_value.keys()]

    z_low, z_high = min(idx_z), max(idx_z)+1  # This plus one is for element to upper node index
    th_low, th_high = min(idx_Th), max(idx_Th)+1

    x_vals_trim = x_vals[th_low: th_high+1]  # This plus one is for Python indexing.
    y_vals_trim = y_vals[z_low: z_high+1]

    # X and Y are boundary points.
    X, Y = numpy.meshgrid(x_vals_trim, y_vals_trim)

    # The values are element centred.
    elem_shape = (X.shape[0]-1, X.shape[1]-1)
    Z = numpy.full(shape=elem_shape, fill_value=numpy.nan)
    Z_ghost = Z.copy()

    for idx, val in elem_idx_to_value.items():
        if idx in active_elements:
            Z[idx.Z-z_low, idx.Th-th_low] = val

        elif INCLUDE_GHOST_ELEMENENTS:
            Z_ghost[idx.Z - z_low, idx.Th - th_low] = val

    all_ghosts_nan = numpy.isnan(Z_ghost).all()

    # Overwrite with the nodes...
    th_range = range(th_low, th_high+1)
    z_range = range(z_low, z_high+1)
    for node_idx, pos in node_idx_pos.items():
        if node_idx.Th in th_range and node_idx.Z in z_range:
            X[node_idx.Z-z_low, node_idx.Th-th_low] = pos.x
            Y[node_idx.Z-z_low, node_idx.Th-th_low] = pos.y

    # Populate with the real values.

    qmesh_real = holoviews.QuadMesh((X, Y, Z), vdims='level', group=contour_view.metric_name)
    qmesh_real.options(cmap='viridis')

    qmesh_list = [qmesh_real]
    if INCLUDE_GHOST_ELEMENENTS and not all_ghosts_nan:
        qmesh_ghost = holoviews.QuadMesh((X, Y, Z_ghost), vdims='level', group=contour_view.metric_name)
        qmesh_ghost.options(cmap='inferno')
        qmesh_list.append(qmesh_ghost)

    for qmesh in qmesh_list:
        qmesh.opts(aspect='equal', line_width=0.1, padding=0.1, width=FIG_SIZE[0], height=FIG_SIZE[1], colorbar=True)

    return holoviews.Overlay(qmesh_list)


class QuadPlateEdge(typing.NamedTuple):
    plate_num: int
    edge_idx: int
    node_num_1: int
    node_num_2: int

    @property
    def node_idx_1(self):
        return self.edge_idx

    @property
    def node_idx_2(self):
        return (self.edge_idx + 1) % 4

    @property
    def global_sorted_nodes(self):
        return tuple(sorted([self.node_num_1, self.node_num_2]))


def get_regular_mesh_data(db: history.DB):
    elem_conn = db.get_element_connections()

    # Step 1 - get all the edges of plates.
    all_edges = []
    min_elem_num = math.inf
    for elem_num, nodes in elem_conn.items():
        if len(nodes) != 4:
            raise ValueError("Quad4 only at the moment")

        min_elem_num = min(min_elem_num, elem_num)
        for edge_idx in range(4):
            node_idx_1 = edge_idx
            node_idx_2 = (edge_idx + 1) % 4
            one_edge = QuadPlateEdge(
                plate_num=elem_num,
                edge_idx=edge_idx,
                node_num_1=nodes[node_idx_1],
                node_num_2=nodes[node_idx_2],
            )

            all_edges.append(one_edge)

    # Step 2 - orientationA and orientationB are perpendicular - one if horizontal and one is vert but we don't know which is which yet.
    sorted_global_node_to_edge = collections.defaultdict(set)
    for one_edge in all_edges:
        sorted_global_node_to_edge[one_edge.global_sorted_nodes].add(one_edge)

    elem_to_edges = collections.defaultdict(set)
    for one_edge in all_edges:
        elem_to_edges[one_edge.plate_num].add(one_edge)

    # These are perpendicular - one if horizontal and one is vert but we don't know which is which yet.
    ORI_A = 0
    ORI_B = 1

    edge_to_ori = dict()

    # Step 3 - populate the known edges by working a "frontier"
    def discard_edge_from_working_set(one_edge):
        sorted_global_node_to_edge[one_edge.global_sorted_nodes].discard(one_edge)
        elem_to_edges[one_edge.plate_num].discard(one_edge)

    # Start off
    arbitrary_edge = elem_to_edges[min_elem_num].pop()
    edge_to_ori[arbitrary_edge] = ORI_A
    discard_edge_from_working_set(arbitrary_edge)
    frontier_edges = {arbitrary_edge,}

    while frontier_edges:
        new_frontier = set()
        for one_edge in frontier_edges:

            # We can tell a relative orientation from the edge index.
            new_edges_from_elem = elem_to_edges[one_edge.plate_num]
            for one_new_edge in new_edges_from_elem:
                same_ori = one_new_edge.edge_idx % 2 == one_edge.edge_idx % 2
                if same_ori:
                    this_ori = edge_to_ori[one_edge]

                else:
                    this_ori = (edge_to_ori[one_edge] + 1) % 2

                edge_to_ori[one_new_edge] = this_ori
                new_frontier.add(one_new_edge)


            # We can also tell when an edge has the same orientation if it has the same global nodes.
            new_edges_from_nodes = sorted_global_node_to_edge[one_edge.global_sorted_nodes]
            for one_new_edge in new_edges_from_nodes:
                this_ori = edge_to_ori[one_edge]
                edge_to_ori[one_new_edge] = this_ori
                new_frontier.add(one_new_edge)

        for one_new_edge in new_frontier:
            discard_edge_from_working_set(one_new_edge)

        frontier_edges.clear()
        frontier_edges.update(new_frontier)


if __name__ == "__main__":
    db_fn = r"E:\Simulations\CeramicBands\v5\pics\4L\history.db"
    with history.DB(db_fn) as db:
        get_regular_mesh_data(db)











