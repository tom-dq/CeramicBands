
import matplotlib.pyplot as plt

import typing

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
    ## TODO  - surface profile - curve fit along the top for simple bending curve.

if __name__ == "__main__":
    db_fn = r"E:\Simulations\CeramicBands\v7\pics\7S\history - Copy.db"
    with history.DB(db_fn) as hist:
        fig, (ax_profile, ax_strain) = plt.subplots(2, 1, sharex=True)
        graph_frame(hist, 650, ax_strain)

        fig.show()
