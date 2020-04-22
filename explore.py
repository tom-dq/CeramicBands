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




def py_vista_test():
    # mesh points
    vertices = numpy.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0],
                         [0, 1, 0],
                         [0.5, 0.5, -1]])

    # mesh faces
    faces = numpy.hstack([[4, 0, 1, 2, 3],  # square
                       [3, 0, 1, 4],  # triangle
                       [3, 1, 2, 4]])  # triangle

    surf = pyvista.PolyData(vertices, faces)

    # plot each face with a different color
    surf.plot(scalars=numpy.arange(3), cpos=[-1, 1, 0.5])


if __name__ == "__main__":
    py_vista_test()
