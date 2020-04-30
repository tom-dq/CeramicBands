import collections
import itertools
import math

import datashader
import numpy
import typing
import holoviews
import panel

from holoviews.operation.datashader import datashade, shade, dynspread, rasterize

import history


FIG_SIZE_UNI = (2000, 1350)
FIG_SIZE_LAPTOP = (1200, 750)

FIG_SIZE = FIG_SIZE_UNI



#hv.Image((range(10), range(10), np.random.rand(10, 10)), datatype=['xarray']).data.hvplot.image('x', 'y', label='test').options(height=1000)


holoviews.extension('bokeh')

class ContourView(typing.NamedTuple):
    db_case_num: int
    contour_key: history.ContourKey

    @property
    def title(self) -> str:
        return f"Case {self.db_case_num}: {self.contour_key.name}"


def make_quadmesh(
        db: history.DB,
        contour_view: ContourView,
) -> holoviews.QuadMesh:
    """Make a single Quadmesh representation"""

    struct_data = get_regular_mesh_data(db)

    node_skeleton = history.NodePosition._all_nones()._replace(result_case_num=contour_view.db_case_num)
    node_coords = {node_pos.node_num: node_pos for node_pos in db.get_all_matching(node_skeleton)}

    # Node positions
    X = numpy.zeros(shape=struct_data.node_indices.shape)
    Y = numpy.zeros(shape=struct_data.node_indices.shape)

    for idx_x, along_y in enumerate(struct_data.node_indices):
        for idx_y, node_num in enumerate(along_y):
            X[idx_x, idx_y] = node_coords[node_num].x
            Y[idx_x, idx_y] = node_coords[node_num].y

    # The values are element centred.
    Z = numpy.zeros(shape=struct_data.elem_indices.shape)



    contour_val_skeleton = history.ContourValue._all_nones()._replace(result_case_num=contour_view.db_case_num, contour_key_num=contour_view.contour_key.value)
    contour_vals = {contour_val.elem_num: contour_val.value for contour_val in db.get_all_matching(contour_val_skeleton)}

    # TEMP - to get rasterize to work, need to make this node-centred.
    Z_nodal_shape = (4,) + struct_data.node_indices.shape
    Z_nodal = numpy.empty(shape = Z_nodal_shape)
    Z_nodal[:] = numpy.nan

    for idx_x, along_y in enumerate(struct_data.elem_indices):
        for idx_y, elem_num in enumerate(along_y):
            c_val = contour_vals.get(elem_num, 0.0)
            Z[idx_x, idx_y] = c_val

            for layer, (z_nodal_idx_x, z_nodal_idx_y) in enumerate(itertools.product( (idx_x, idx_x+1), (idx_y, idx_y+1) )):
                Z_nodal[layer, z_nodal_idx_x, z_nodal_idx_y] = c_val

    Z_nodal_flat = numpy.nanmean(Z_nodal, axis=0)


    qmesh = holoviews.QuadMesh((X, Y, Z_nodal_flat), vdims='level', group=contour_view.title)
    qmesh.options(cmap='viridis')

    qmesh.opts(aspect='equal', line_width=0.1, padding=0.1, colorbar=True, width=FIG_SIZE[0], height=FIG_SIZE[1])

    qmesh.data.hvplot.image('x', 'y', label=contour_view.title).options( width=FIG_SIZE[0], height=FIG_SIZE[1])
    return qmesh


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

    def opposing_node(self, n1: int) -> int:
        if n1 == self.node_num_1:
            return self.node_num_2

        elif n1 == self.node_num_2:
            return self.node_num_1

        else:
            raise ValueError(n1)


class StructuredMeshData(typing.NamedTuple):
    node_indices: numpy.ndarray
    elem_indices: numpy.ndarray


def get_regular_mesh_data(db: history.DB) -> StructuredMeshData:
    """Returns a numpy array of node indices for a structured mesh."""
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

    # Step 4 - find a corner
    nodes_on_edges = collections.defaultdict(set)
    for one_edge in all_edges:
        for n in [one_edge.node_num_1, one_edge.node_num_2]:
            nodes_on_edges[n].add(one_edge)

    corner_nodes = [node_num for node_num, edges in nodes_on_edges.items() if len(edges) == 2]

    # Step 5 - send out "streamers" from the corner
    def make_streamer(starting_node: int, ori) -> typing.List[int]:
        """Get the nodes in a line."""

        streamer = [starting_node]
        still_going = True
        while still_going:
            frontier_node = streamer[-1]
            if len(streamer) > 1:
                backwards_node = streamer[-2]

            else:
                backwards_node = None

            next_edges = [edge for edge in nodes_on_edges[frontier_node] if edge_to_ori[edge] == ori and backwards_node not in edge.global_sorted_nodes]
            if not next_edges:
                still_going = False

            else:
                next_edge = next_edges.pop()
                streamer.append(next_edge.opposing_node(frontier_node))

        return streamer

    arbitrary_corner_node = min(corner_nodes)
    ori_to_streamer = {ori: make_streamer(arbitrary_corner_node, ori) for ori in [ORI_A, ORI_B]}

    len_to_ori = {len(streamer): ori for ori, streamer in ori_to_streamer.items()}
    X_ORI = max(len_to_ori.items())[1]
    Y_ORI = (X_ORI + 1) % 2

    # Step 6 - Make the node indices for the structured mesh
    node_indices = numpy.zeros( shape=(len(ori_to_streamer[X_ORI]), len(ori_to_streamer[Y_ORI])), dtype=int )

    for y_idx, y_starting_node in enumerate(ori_to_streamer[Y_ORI]):
        x_streamer = make_streamer(y_starting_node, X_ORI)
        node_indices[:, y_idx] = x_streamer


    # Step 7 - element indices are determined based on the 2x2 patch and the nodes.
    sorted_nodes_to_elem = {tuple(sorted(conn)): elem_num for elem_num, conn in elem_conn.items()}
    elem_shape = (node_indices.shape[0] - 1, node_indices.shape[1] - 1)
    elem_indices = numpy.zeros( shape=elem_shape, dtype=int)

    x_idxs = range(node_indices.shape[0]-1)
    y_idxs = range(node_indices.shape[1]-1)
    for x_node_idx, y_node_idx in itertools.product(x_idxs, y_idxs):
        window = node_indices[x_node_idx:x_node_idx+2, y_node_idx:y_node_idx+2]
        sorted_nodes = tuple(sorted(window.flatten()))
        elem_num = sorted_nodes_to_elem[sorted_nodes]
        elem_indices[x_node_idx, y_node_idx] = elem_num

    return StructuredMeshData(
        node_indices=node_indices,
        elem_indices=elem_indices,
    )


def make_dashboard(db: history.DB):

    cases = db.get_all(history.ResultCase)
    all_views = [ContourView(db_case_num=case.num, contour_key=history.ContourKey.prestrain_mag) for case in cases]

    plot_dict = {}
    for contour_view in all_views:
        qmesh = make_quadmesh(db, contour_view)
        #plot_dict[contour_view] = rasterize(qmesh)

    # hmap = holoviews.HoloMap(plot_dict, kdims=list(contour_view._fields))

    ds_mesh = datashade(qmesh)
    ds_mesh.opts(plot=dict(width=500, height=500))
    ds_mesh.options(width=1000, height=500)
    p = panel.panel(ds_mesh)

    p.show()




if __name__ == "__main__":
    db_fn = r"E:\Simulations\CeramicBands\v5\pics\54\history - Copy.db"
    with history.DB(db_fn) as db:
        make_dashboard(db)
        input()











