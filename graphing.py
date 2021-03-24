import itertools

import matplotlib.pyplot as plt

import typing

import st7

import history

def graph_frame(hist: history.DB, db_res_case: int, ax: plt.Axes):
    row_skeleton = history.ColumnResult._all_nones()._replace(result_case_num=db_res_case)
    column_data = list(hist.get_all_matching(row_skeleton))

    col_data_graph = [cd for cd in column_data if cd.contour_key == history.ContourKey.total_strain_x]

    for yielded, base_col in (
            (False, 'tab:blue'),
            (True, 'tab:orange'),
    ):
        # Fall back to NaNs, only override with the real data.
        x_to_cd = {cd.x: history.ColumnResult._nans_at(cd.x) for cd in col_data_graph}

        # Override with the real thing
        for cd in col_data_graph:
            if cd.yielded == yielded:
                x_to_cd[cd.x] = cd

        res_to_plot = [cd for x, cd in sorted(x_to_cd.items())]
        x = [cd.x for cd in res_to_plot]
        y_min = [cd.minimum for cd in res_to_plot]
        y_mean = [cd.mean for cd in res_to_plot]
        y_max = [cd.maximum for cd in res_to_plot]

        ax.fill_between(x, y_min, y_max, color=base_col, alpha=0.2)
        ax.plot(x, y_mean, color=base_col)


def graph_surface_profile(hist: history.DB, db_res_case: int, ax: plt.Axes):
    pass
    ## TODO  - surface profile - curve fit along the top for simple bending_pure curve.


def graph_force_disp(hist: history.DB, ax: plt.Axes):

    fd_data = list(hist.get_all(history.LoadDisplacementPoint))

    # Put marks on the final increment of each load step
    res_case_data = list(hist.get_all(history.ResultCase))
    res_case_data.sort()
    maj_to_max_res_case = dict()
    for res_case in res_case_data:
        maj_to_max_res_case[res_case.major_inc] = res_case.num

    final_minor_inc_cases = set(maj_to_max_res_case.values())

    def sort_key(ld_point: history.LoadDisplacementPoint):
        return ld_point.node_num, ld_point.load_text, ld_point.disp_text

    fd_data.sort(key=sort_key)

    for (node_num, load_text, disp_text), fd_points in itertools.groupby(fd_data, sort_key):
        fd_points = list(fd_points)
        x = [p.disp_val for p in fd_points]
        y = [p.load_val for p in fd_points]

        show_marker = [p.result_case_num in final_minor_inc_cases for p in fd_points]

        ax.plot(x, y, marker='x', markevery=show_marker, label=f"{node_num} {disp_text} vs {load_text}")

        # Hacky getting DoF
        dof_iter = (dof for dof in st7.DoF if dof.rx_mz_text == load_text)
        dof = next(dof_iter)

        ax.set_xlabel(dof.disp_or_rotation_text)
        ax.set_ylabel(dof.force_or_moment_text)


    ax.legend()



if __name__ == "__main__":
    db_fn = r"E:\Simulations\CeramicBands\v7\pics\8L\history - Copy.db"
    with history.DB(db_fn) as hist:
        fig, (ax_fd, ax_strain,) = plt.subplots(2, 1)
        graph_frame(hist, 650, ax_strain)
        graph_force_disp(hist, ax_fd)

    fig.show()
