import itertools

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter as ThisWriter

import typing

import st7

import history

def graph_frame(hist: history.DB, db_res_case: int, ax: plt.Axes):
    row_skeleton = history.ColumnResult._all_nones()._replace(result_case_num=db_res_case)
    column_data = list(hist.get_all_matching(row_skeleton))

    col_data_graph = [cd for cd in column_data if cd.contour_key == history.ContourKey.total_strain_x]

    ax.clear()
    graphed_something = False

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
                graphed_something = True

        res_to_plot = [cd for x, cd in sorted(x_to_cd.items())]
        x = [cd.x for cd in res_to_plot]
        y_min = [cd.minimum for cd in res_to_plot]
        y_mean = [cd.mean for cd in res_to_plot]
        y_max = [cd.maximum for cd in res_to_plot]

        ax.fill_between(x, y_min, y_max, color=base_col, alpha=0.2)
        ax.plot(x, y_mean, color=base_col)

    return graphed_something

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

def _add_margin(one_range):
    diff = one_range[1] - one_range[0]
    margin = 0.05 * diff

    return one_range[0] - margin, one_range[1] + margin


def animate_test():
    db_fn = r"E:\Simulations\CeramicBands\v7\pics\8T\history.db"
    movie_writer = ThisWriter(fps=30)
    fig, ax = plt.subplots(1, 1, figsize=(20, 10), constrained_layout=True)
    # fig.tight_layout()
    with movie_writer.saving(fig, r'E:\TEMP\8T.mp4', dpi=300):
        with history.DB(db_fn) as hist:
            y_range = hist.get_column_result_range()
            y_range_padded = _add_margin(y_range)

            for db_res_case in sorted(hist.get_all(history.ResultCase)):
                print(db_res_case)

                graphed_something = graph_frame(hist, db_res_case.num, ax)

                ax.set_ylim(y_range_padded)
                fig.canvas.draw()
                fig.canvas.flush_events()
                if graphed_something:
                    movie_writer.grab_frame()

if __name__ == "__main__":
    animate_test()

if False:
    db_fn = r"E:\Simulations\CeramicBands\v7\pics\8T\history.db"
    with history.DB(db_fn) as hist:
        fig, (ax_fd, ax_s1, ax_s2, ax_s3) = plt.subplots(4, 1)
        graph_frame(hist, 100, ax_s1)
        graph_frame(hist, 700, ax_s2)
        graph_frame(hist, 1200, ax_s3)
        graph_force_disp(hist, ax_fd)

    fig.show()
