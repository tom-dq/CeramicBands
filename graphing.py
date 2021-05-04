import itertools
import pathlib
import glob
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter as ThisWriter

from scipy.optimize import curve_fit

import typing

import st7

import history


size_4k = (3840, 2160)

DPI_FIG = 200

class CurveCoefficients(typing.NamedTuple):
    a: float
    b: float
    c: float

    def __call__(self, x: float) -> float:
        return self.a * x**2 + self.b * x + self.c

    def make_nice_name(self) -> str:
        return f"{self.a:1.4g} x^2 + {self.b:1.4g} x + {self.c:1.4g}"

    @staticmethod
    def p0_guess():
        return CurveCoefficients(a=0.001, b=0.1, c=1)

    @staticmethod
    def curve_name():
        return "Quadratic"



def graph_frame(hist: history.DB, db_res_case: int, contour_key: history.ContourKey, ax: plt.Axes):
    row_skeleton = history.ColumnResult._all_nones()._replace(result_case_num=db_res_case)
    column_data = list(hist.get_all_matching(row_skeleton))

    col_data_graph = [cd for cd in column_data if cd.contour_key == contour_key]

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
        yield_text = "Dilated" if yielded else "Undilated"
        ax.plot(x, y_mean, color=base_col, label=f"{contour_key.name} ({yield_text})")

    ax.legend()

    return graphed_something


def _node_of_top_line(hist: history.DB) -> typing.FrozenSet[int]:

    row_skeleton = history.NodePosition._all_nones()._replace(result_case_num=1)
    node_first_increment = list(hist.get_all_matching(row_skeleton))
    if not node_first_increment:
        return frozenset()

    top_y = max(n.y for n in node_first_increment)

    EPS = 1e-6
    top_row = [n for n in node_first_increment if abs(n.y - top_y) <= EPS]

    return frozenset(n.node_num for n in top_row)



def _get_surface_curve(nodes_top_line, hist: history.DB, db_res_case: int) -> CurveCoefficients:



    def func(x, *args):
        curve_coeffs = CurveCoefficients(*args)
        return curve_coeffs(x)

    x_data = [n.x for n in nodes_top_line]
    y_data = [n.y for n in nodes_top_line]

    popt, pconv = curve_fit(func, x_data, y_data, p0=CurveCoefficients.p0_guess())

    curve_floats = [float(x) for x in popt]
    curve_coeffs = CurveCoefficients(*curve_floats)

    return curve_coeffs


class LineData(typing.NamedTuple):
    x_data: typing.Iterable[float]
    y_data: typing.Iterable[float]
    colour: str
    label: str
    linestyle: str

    def y_within_bounds(self, x_lims) -> typing.Iterable[float]:
        x_min, x_max = x_lims
        for x, y in itertools.zip_longest(self.x_data, self.y_data):
            if x_min <= x <= x_max:
                yield y





def _get_graph_surface_profile_data(deviation_only: bool, top_nodes: typing.FrozenSet[int], hist: history.DB, db_res_case: int) -> typing.Iterable[LineData]:

    row_skeleton = history.NodePosition._all_nones()._replace(result_case_num=db_res_case)
    nodes_this_increment = list(hist.get_all_matching(row_skeleton))
    nodes_top_line = [n for n in nodes_this_increment if n.node_num in top_nodes]

    top_curve = _get_surface_curve(nodes_top_line, hist, db_res_case)

    def sort_key(n):
        return n.x

    nodes_top_line.sort(key=sort_key)

    x_data = [n.x for n in nodes_top_line]

    if deviation_only:
        y_deviation = [n.y - top_curve(n.x) for n in nodes_top_line]
        yield LineData(x_data, y_deviation, 'black', f"Top Surface Deviation from {CurveCoefficients.curve_name()}", '-')

    else:
        y_data = [n.y for n in nodes_top_line]
        y_fit = [top_curve(x) for x in x_data]

        yield LineData(x_data, y_fit, 'tab:gray', f"{CurveCoefficients.curve_name()} best fit: {top_curve.make_nice_name()}", '--')
        yield LineData(x_data, y_data, 'black', f"Top Surface", '-')


def _get_overall_limits(deviation_only: bool, top_nodes: typing.FrozenSet[int], hist: history.DB, x_lims: typing.Tuple[float, float]) -> typing.Tuple[float, float]:

    all_y_points = []
    for db_res_case in sorted(hist.get_all(history.ResultCase)):
        for line_data in _get_graph_surface_profile_data(deviation_only, top_nodes, hist, db_res_case.num):
            these_y_points = line_data.y_within_bounds(x_lims)
            all_y_points.extend(these_y_points)

    return min(all_y_points), max(all_y_points)


def graph_surface_profile(deviation_only: bool, top_nodes: typing.FrozenSet[int], hist: history.DB, db_res_case: int, ax: plt.Axes):

    ax.clear()

    for line_data in _get_graph_surface_profile_data(deviation_only, top_nodes, hist, db_res_case):
        ax.plot(line_data.x_data, line_data.y_data, color=line_data.colour, label=line_data.label, linestyle=line_data.linestyle)

    ax.legend()


def graph_force_disp(hist: history.DB, db_res_case, ax: plt.Axes):

    ax.clear()

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

        # Point to show current db res case
        x_one = [p.disp_val for p in fd_points if p.result_case_num == db_res_case]
        y_one = [p.load_val for p in fd_points if p.result_case_num == db_res_case]

        ax.plot(x_one, y_one, marker='o', markeredgecolor='tab:orange', markerfacecolor='tab:orange')


    ax.legend()

def _add_margin(one_range):
    diff = one_range[1] - one_range[0]
    margin = 0.02 * diff

    return one_range[0] - margin, one_range[1] + margin


class SubGraphs(typing.NamedTuple):
    force_disp: typing.Optional[matplotlib.pyplot.Axes]
    surface_absolute: typing.Optional[matplotlib.pyplot.Axes]
    surface_deviation: typing.Optional[matplotlib.pyplot.Axes]
    columns_strain_x: typing.Optional[matplotlib.pyplot.Axes]

    def num_graphs(self) -> int:
        return sum(1 for x in self if x)

    def any_with_common_x(self) -> bool:
        return any((
            self.surface_absolute,
            self.surface_deviation,
            self.columns_strain_x,
        ))

def create_subplots(sub_graph_flags: SubGraphs) -> typing.Tuple[matplotlib.pyplot.Figure, SubGraphs]:
    fig_size_mpl = [x / DPI_FIG for x in size_4k]

    num_graphs = sub_graph_flags.num_graphs()
    fig, ax = plt.subplots(num_graphs, 1, figsize=fig_size_mpl, dpi=100, constrained_layout=True)

    if num_graphs == 1:
        ax_list = [ax]

    else:
        ax_list = ax

    iter_ax = iter(ax_list)

    working_dict = {}
    for key, flag in sub_graph_flags._asdict().items():
        this_ax = next(iter_ax) if flag else None

        working_dict[key] = this_ax

    return fig, SubGraphs(**working_dict)



def animate_movie(db_fn: str, graph_movie_fn: str):
    # db_fn = r"E:\Simulations\CeramicBands\v7\pics\8X\history.db"
    movie_writer = ThisWriter(fps=30)

    sub_graph_flags = SubGraphs(force_disp=True, surface_absolute=False, surface_deviation=True, columns_strain_x=True)
    fig, sub_graphs = create_subplots(sub_graph_flags)

    # fig.tight_layout()
    with movie_writer.saving(fig, graph_movie_fn, dpi=DPI_FIG):
        with history.DB(db_fn) as hist:
            top_nodes = _node_of_top_line(hist)

            if not top_nodes:
                # Empty DB - get out of here.
                return

            lims = get_limits_graphs(hist, top_nodes, sub_graphs)

            for db_res_case in sorted(hist.get_all(history.ResultCase)):
                print(db_res_case)

                graphed_something = compose_graphs(hist, top_nodes, db_res_case.num, sub_graphs)

                # Set any global limits which have been assigned
                for ax in sub_graphs:
                    if ax and ax in lims:
                        x_lims, y_lims = lims[ax]

                        ax.set_xlim(x_lims)
                        ax.set_ylim(y_lims)

                fig.canvas.draw()
                fig.canvas.flush_events()
                if graphed_something:
                    movie_writer.grab_frame()


def get_limits_graphs(hist: history.DB, top_nodes, sub_graphs: SubGraphs):
    """Make a dict of ax to x and y limits"""

    # x limits come from the column graph
    if sub_graphs.any_with_common_x():
        column_data = list(hist.get_all(history.ColumnResult))
        x_vals = [cd.x for cd in column_data]
        x_lims = min(x_vals), max(x_vals)

    raw_y_range = {}
    if sub_graphs.surface_absolute:
        raw_y_range[sub_graphs.surface_absolute] = _get_overall_limits(False, top_nodes, hist, x_lims)

    if sub_graphs.surface_deviation:
        raw_y_range[sub_graphs.surface_deviation] = _get_overall_limits(True, top_nodes, hist, x_lims)

    if sub_graphs.columns_strain_x:
        raw_y_range[sub_graphs.columns_strain_x] = hist.get_column_result_range(history.ContourKey.total_strain_x)

    out_d = {
        ax: (x_lims, _add_margin(y_range)) for ax, y_range in raw_y_range.items()
    }

    return out_d



def compose_graphs(hist: history.DB, top_nodes, db_res_case, sub_graphs: SubGraphs):
    # ax_fd, ax_s1, ax_s2, ax_s3 = ax_list

    if sub_graphs.surface_absolute:
        graph_surface_profile(False, top_nodes, hist, db_res_case, sub_graphs.surface_absolute)

    if sub_graphs.surface_deviation:
        graph_surface_profile(True, top_nodes, hist, db_res_case, sub_graphs.surface_deviation)

    if sub_graphs.columns_strain_x:
        graphed_something_x = graph_frame(hist, db_res_case, history.ContourKey.total_strain_x, sub_graphs.columns_strain_x)

    else:
        graphed_something_x = True

    if sub_graphs.force_disp:
        graph_force_disp(hist, db_res_case, sub_graphs.force_disp)

    return graphed_something_x


def make_graph_in_working_dir(working_dir: str):
    working_dir = pathlib.Path(working_dir)

    db_fn = working_dir / "history.db"
    graph_movie_file = f"graphs-{working_dir.parts[-1]}-with-fd.mp4"
    graph_movie_fn = working_dir / graph_movie_file

    if not graph_movie_fn.is_file():
        animate_movie(str(db_fn), str(graph_movie_fn))
        return f"Finished {graph_movie_fn}"

    else:
        return f"Skipped  {graph_movie_fn}"


def make_all_graph_movies_mp():
    N_WORKERS=14

    dirs_to_do = glob.glob(r"E:\Simulations\CeramicBands\v7\pics\[9-A]*")

    with multiprocessing.Pool(N_WORKERS) as pool:
        for x in pool.imap_unordered(make_graph_in_working_dir, dirs_to_do):
            print(x)


def make_all_graph_movies():
    db_fn = r"E:\Simulations\CeramicBands\v7\pics\9K\history.db"
    graph_movie_fn = r"E:\Simulations\CeramicBands\v7\pics\9K\graphs-9K-bbb.mp4"

    animate_movie(db_fn, graph_movie_fn)


if __name__ == "__main__":
    # make_all_graph_movies()
    pass

if __name__ == "__main__":
    db_fn = r"E:\Simulations\CeramicBands\v7\pics\98\history.db"

    sub_graph_flags = SubGraphs(force_disp=True, surface_absolute=False, surface_deviation=False, columns_strain_x=False)
    fig, sub_graphs = create_subplots(sub_graph_flags)

    with history.DB(db_fn) as hist:
        top_nodes = _node_of_top_line(hist)

        compose_graphs(hist, top_nodes, 1000, sub_graphs)

    fig.show()
    input()
