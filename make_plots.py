import functools
import pathlib
import statistics
import typing
import os
import enum
import itertools

import collections
import hashlib
import csv


import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.markers
import matplotlib.cm
from cycler import cycler

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from PIL import Image

from tables import Table
import common_types
import image_cropper

from config import active_config

import main
import history
import annotation_tile
import aletha_csv

# To get pickle to unpickle
from main import CheckpointState, save_state
from main import RunParams
from main import ModelFreedomCase
from main import Ratchet
from main import PrestrainUpdate
from main import ResultFrame

_keep_these_in_for_pickling_ = [
    CheckpointState,
    RunParams,
    ModelFreedomCase,
    Ratchet,
    PrestrainUpdate,
    ResultFrame,
]

T_Path = typing.Union[pathlib.Path, str]

# plot_data_base=pathlib.Path(r"C:\Users\Tom Wilson\Dropbox\PhD\Ceramic Bands Source Models\Test Output")
plot_data_base = pathlib.Path(
    r"C:\Users\Tom Wilson\Documents\CeramicBandData\outputs\192.168.1.109+8080\v7\pics"
)
graph_output_base = pathlib.Path(
    r"C:\Users\Tom Wilson\Dropbox\PhD\Papers\Mike-1-Ceramic Bands\2023-v1"
)

# For comparison with the physical specimen, adjust some graphs with these quantities...
SPECIMEN_NOMINAL_LENGTH_MM = 21.0
FULL_BEAM_HEIGHT = 3.0


def friendly_str(x: typing.Union[str, int, float]) -> str:
    if isinstance(x, float):
        return f"{x:g}"

    return str(x)


def get_last_case_image_fn(path: pathlib.Path) -> pathlib.Path:
    """Get the image file from the directory"""

    images = list(path.glob("Case-*.png"))
    if len(images) != 1:
        raise ValueError(images)

    return images[0]


class NoResultException(Exception):
    pass


class ResultWithError(typing.NamedTuple):
    val: float
    y_sd: typing.Optional[float]

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            match self.y_sd:
                case float():
                    new_y_sd = other * self.y_sd

                case int():
                    new_y_sd = other * self.y_sd

                case None:
                    new_y_sd = None

                case _:
                    raise TypeError(self.y_sd)

            return ResultWithError(val=other * self.val, y_sd=new_y_sd)

        return NotImplemented

    def __rmul__(self, other):
        # Hacky I'm sorry
        return self.__mul__(other)


class BandSizeRatio(typing.NamedTuple):
    run_params: RunParams
    bands: typing.List[history.TransformationBand]
    result_case_num: int

    def _abs_band_sizes(self) -> typing.List[float]:
        band_sizes = [abs(b.band_size) for b in self.bands]

        return sorted(band_sizes)

    def get_ratios(self) -> typing.List[float]:
        band_sizes = self._abs_band_sizes()

        min_abs_size = min(band_sizes)
        scaled = [b / min_abs_size for b in band_sizes]
        return sorted(scaled)

    def get_nominal_lower_size(self) -> float:
        """The nominal "min" size of a band, taken as the 10th percentile"""
        quantiles = statistics.quantiles(self._abs_band_sizes(), n=10)
        return quantiles[0]

    def get_nominal_upper_size(self) -> float:
        """The nominal "max" size of a band, taken as the 90th percentile"""
        quantiles = statistics.quantiles(self._abs_band_sizes(), n=10)
        return quantiles[-1]

    def get_scale(self) -> float:
        return self.major_band_threshold()

    def major_band_threshold(self) -> float:

        # TODO - fine tuning this...
        bs_min = self.get_nominal_lower_size()
        bs_max = self.get_nominal_upper_size()
        bs_diff = bs_max - bs_min

        bs_prop = bs_min + 1 / 8 * bs_diff
        return bs_prop

    def get_major_band_count_ratio(self) -> ResultWithError:
        """Proportion of the transformation bands which are "major" bands."""

        cutoff = self.major_band_threshold()

        maj_bands = [bs for bs in self._abs_band_sizes() if bs > cutoff]
        val = len(maj_bands) / len(self.bands)

        return ResultWithError(val=val, y_sd=None)

    def get_major_band_spacing(self) -> float:
        """Average distance between major bands."""

        cutoff = self.major_band_threshold()

        maj_bands = [b for b in self.bands if abs(b.band_size) > cutoff]

        maj_band_x_vals = sorted(bs.x for bs in maj_bands)

        n_band_gaps = len(maj_bands) - 1
        maj_band_span = maj_band_x_vals[-1] - maj_band_x_vals[0]

        return maj_band_span / n_band_gaps

    def get_major_band_spacing_sd(self) -> ResultWithError:
        """Get the mean and SD, to match the A. Liens paper. e.g.:
        However, for the
        3 mm-thick samples, a very large SD is observed, leading to the
        maximum bands' width as large as 120 μm (Figs. 8c and 11)."""

        cutoff = self.major_band_threshold()

        maj_bands = [b for b in self.bands if abs(b.band_size) > cutoff]

        maj_band_x_vals = sorted(bs.x for bs in maj_bands)

        def pairwise(vals):
            vals1, vals2 = itertools.tee(vals, 2)
            _ = next(vals2)
            yield from zip(vals1, vals2)

        spacings = [x2 - x1 for x1, x2 in pairwise(maj_band_x_vals)]

        spacing_mean = statistics.mean(spacings)
        spacing_sd = statistics.stdev(spacings, xbar=spacing_mean)

        return ResultWithError(val=spacing_mean, y_sd=spacing_sd)

    def get_band_and_maj_ratio(
        self,
    ) -> typing.Iterable[typing.Tuple[float, history.TransformationBand]]:
        """Returns the bands in order, with the ratio of the band size. So if larger than one, it's a major band."""

        cutoff = self.major_band_threshold()

        for band in sorted(self.bands):
            yield abs(band.band_size) / cutoff, band

    def get_num_maj_bands_full_length(self) -> ResultWithError:
        """Assume the major band density seen in the current range is replicated along the full length"""

        maj_band_spacing = self.get_major_band_spacing()
        val = SPECIMEN_NOMINAL_LENGTH_MM / maj_band_spacing

        return ResultWithError(val=val, y_sd=None)

    def get_beam_depth(self) -> float:

        return FULL_BEAM_HEIGHT * self.run_params.scale_model_y


def make_example_table() -> Table:
    STRESS_START = 400
    stress_end = 450
    dilation_ratio = 0.008

    prestrain_table = Table(
        [
            common_types.XY(0.0, 0.0),
            common_types.XY(STRESS_START, 0.0),
            common_types.XY(stress_end, -1 * dilation_ratio),
            common_types.XY(stress_end + 200, -1 * dilation_ratio),
        ]
    )

    return prestrain_table


def show_table(table: Table):

    x_vals = [p.x for p in table.data]
    y_vals = [-100 * p.y for p in table.data]

    ax = plt.gca()

    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())

    plt.plot(x_vals, y_vals)
    plt.title("Dilation with stress ratchet")
    plt.xlabel("Stress (MPa)")
    plt.ylabel("Dilation")

    plt.xlim((380, 470))
    plt.show()


def _get_last_result_case_num(
    saved_state: CheckpointState,
    cases: typing.List[history.ResultCase],
    accept_not_the_last_increment: bool,
) -> history.ResultCase:
    if accept_not_the_last_increment:
        return cases[-1]

    last_maj_inc = saved_state.run_params.get_maj_incs(one_based=True)[-1]

    if any(c.major_inc > last_maj_inc for c in cases):
        raise ValueError(
            f"Logical error - only expected to have up to case {last_maj_inc}"
        )

    cases_subset = [c for c in cases if c.major_inc == last_maj_inc]
    minor_incs = [c.minor_inc for c in cases_subset]

    if not minor_incs:
        raise NoResultException(minor_incs)

    last_minor_inc = max(minor_incs)

    one_case = [c for c in cases_subset if c.minor_inc == last_minor_inc]

    if len(one_case) != 1:
        raise NoResultException(one_case)

    return one_case.pop()


@functools.lru_cache(maxsize=1024)
def make_band_min_maj_comparison(
    working_dir: T_Path, accept_not_the_last_increment: bool
) -> BandSizeRatio:

    working_dir = pathlib.Path(working_dir)
    try:
        saved_state = main.load_state(working_dir)
    except FileNotFoundError:

        raise NoResultException()

    with history.DB(working_dir / "history.db") as db:
        cases = list(db.get_all(history.ResultCase))
        last_case = _get_last_result_case_num(
            saved_state, cases, accept_not_the_last_increment
        )

        print(
            working_dir.parts[-1],
            last_case._replace(name=""),
            saved_state.run_params.scale_model_y,
            saved_state.run_params.scaling._spacing,
        )

        band_skeleton = history.TransformationBand._all_nones()._replace(
            result_case_num=last_case.num
        )

        last_case_bands = list(db.get_all_matching(band_skeleton))

    return BandSizeRatio(
        run_params=saved_state.run_params,
        bands=last_case_bands,
        result_case_num=last_case.num,
    )


class DataSource(enum.Enum):
    single_study_single_point = enum.auto()
    single_band_single_point = enum.auto()


class DataSource(enum.Enum):
    single_study_single_point = enum.auto()
    single_band_single_point = enum.auto()


class PlotType(enum.Enum):
    maj_ratio = "Proportion of major transformation bands"
    maj_spacing = "Average spacing between major bands"
    num_bands = f"Number of bands over {SPECIMEN_NOMINAL_LENGTH_MM} mm length"
    band_aspect_ratio = "Band depth/thickness comparison"

    def get_y_axis_limits(self) -> typing.Optional[typing.Tuple[float, float]]:
        if self == PlotType.maj_spacing:
            #return (80.0, 320.0,)
            return (0.0, 650.0,)

        elif self == PlotType.num_bands:
            return (
                60.0,
                180.0,
            )

        elif self == PlotType.maj_ratio:
            return (
                0.25,
                0.65,
            )

        elif self == PlotType.band_aspect_ratio:
            return (0.0, 0.1)

        else:
            raise ValueError(self)

    def get_data_source(self) -> DataSource:
        if self == PlotType.band_aspect_ratio:
            return DataSource.single_band_single_point

        elif self in {PlotType.maj_ratio, PlotType.maj_spacing, PlotType.num_bands}:
            return DataSource.single_study_single_point

        elif self == PlotType.band_aspect_ratio:
            return (0.0, 0.1)

        else:
            raise ValueError(self)

    def get_data_source(self) -> DataSource:
        if self == PlotType.band_aspect_ratio:
            return DataSource.single_band_single_point

        elif self in {PlotType.maj_ratio, PlotType.maj_spacing, PlotType.num_bands}:
            return DataSource.single_study_single_point

        else:
            raise ValueError(self)


class XAxis(enum.Enum):
    beam_depth = enum.auto()
    dilation_max = enum.auto()
    run_index = enum.auto()  # Just order the simulations one by one.
    initiation_variation = enum.auto()
    initiation_spacing = enum.auto()
    band_depth_ratio = enum.auto()

    def get_x_label(self) -> str:
        d = {
            XAxis.beam_depth: "Beam Depth (mm)",
            XAxis.dilation_max: "Dilation (Max)",
            XAxis.run_index: "Run Index",
            XAxis.initiation_variation: "Initiation Variation",
            XAxis.initiation_spacing: "Initiation Spacing (mm)",
            XAxis.band_depth_ratio: "Band Depth / Thickenss",
        }

        return d[self]

    def get_x_range(self) -> typing.Optional[typing.Tuple[float, float]]:
        if self == XAxis.beam_depth:
            return (0.0, 3.5)

        elif self == XAxis.band_depth_ratio:
            return (0.0, 0.6)

        return None

    def get_legend_label(self) -> str:
        d = {
            XAxis.beam_depth: "BeamDepth",
            XAxis.dilation_max: "DilationRatio",
            XAxis.run_index: "Run",
            XAxis.initiation_variation: "InitiationVariation",
            XAxis.initiation_spacing: "InitiationSpacing",
            XAxis.band_depth_ratio: "DepthRatio",
        }

        return d[self]

    def get_applicable_plot_types(self) -> typing.Iterable[PlotType]:
        aspect_plot_types = {
            PlotType.band_aspect_ratio,
        }
        if self == XAxis.band_depth_ratio:
            # Special case - points on graph all together
            return aspect_plot_types

        else:
            # Normal case - one study is one point on the graph.
            normal_plot_types = [pt for pt in PlotType if pt not in aspect_plot_types]
            return normal_plot_types


class Study(typing.NamedTuple):
    name: str
    band_size_ratios: typing.List[BandSizeRatio]
    x_axis: XAxis
    images_to_annotate: typing.Set[str]
    tile_position: annotation_tile.TilePosition


def _generate_study_data(
    name,
    x_axis,
    images_to_annotate: typing.Optional[typing.Set[str]],
    tile_position: annotation_tile.TilePosition,
    gen_relevant_subdirectories,
    accept_not_the_last_increment: bool,
) -> Study:
    # Get the relevant subdirectories for inclusion

    if images_to_annotate is None:
        images_to_annotate = set()

    def make_bsrs():
        for working_dir in gen_relevant_subdirectories():
            try:
                band_size_ratio = make_band_min_maj_comparison(
                    working_dir, accept_not_the_last_increment
                )
                if band_size_ratio.result_case_num > 800:
                    yield band_size_ratio

            except NoResultException:
                pass

    return Study(
        name=name,
        band_size_ratios=list(make_bsrs()),
        x_axis=x_axis,
        images_to_annotate=images_to_annotate,
        tile_position=tile_position,
    )


def generate_plot_data_range(
    name,
    x_axis,
    first_considered_subdir: str,
    last_considered_subdir: str,
    tile_position: annotation_tile.TilePosition,
    images_to_annotate: typing.Optional[typing.Set[str]] = None,
    accept_not_the_last_increment: bool = False,
) -> Study:
    def gen_relevant_subdirectories():
        min_hex = int(first_considered_subdir, base=36)
        max_hex = int(last_considered_subdir, base=36)

        for working_dir in plot_data_base.iterdir():
            if working_dir.is_dir():
                this_hex = int(working_dir.parts[-1], base=36)
                if min_hex <= this_hex <= max_hex:
                    yield working_dir

    return _generate_study_data(
        name,
        x_axis,
        images_to_annotate,
        tile_position,
        gen_relevant_subdirectories,
        accept_not_the_last_increment,
    )


def generate_plot_data_specified(
    name,
    x_axis,
    dir_ends: typing.List[str],
    tile_position: annotation_tile.TilePosition,
    images_to_annotate: typing.Optional[typing.Set[str]] = None,
    accept_not_the_last_increment: bool = False,
) -> Study:
    def gen_relevant_subdirectories():
        for de in dir_ends:
            yield plot_data_base / de

    return _generate_study_data(
        name,
        x_axis,
        images_to_annotate,
        tile_position,
        gen_relevant_subdirectories,
        accept_not_the_last_increment,
    )


def _fig_fn(plot_type: typing.Optional[PlotType] = None):
    name_end = plot_type.name if plot_type else "depth_compare"
    return graph_output_base / f"{name_end}.png"


def _bsr_list_hash(bsr_list: typing.List[BandSizeRatio]) -> str:
    """Unique string for the output plots so they don't overwrite each other."""

    dir_ends = [bsr.run_params.working_dir.name for bsr in bsr_list]
    hash_bit = hashlib.md5("-".join(dir_ends).encode()).hexdigest()[0:6]

    return f"{len(bsr_list)}-{hash_bit}"


def get_x_axis_val_raw(study: Study, bsr: BandSizeRatio):
    if study.x_axis == XAxis.beam_depth:
        return bsr.get_beam_depth()

    elif study.x_axis == XAxis.initiation_spacing:
        return bsr.run_params.scaling._spacing

    elif study.x_axis == XAxis.initiation_variation:
        return bsr.run_params.scaling._max_variation

    elif study.x_axis == XAxis.run_index:
        return bsr.run_params.working_dir.name

    elif study.x_axis == XAxis.dilation_max:
        return bsr.run_params.parameter_trend.dilation_ratio.get_single_value_returned()

    elif study.x_axis == XAxis.band_depth_ratio:
        # This is not actually the x axis but it's not used that way for this plot type.
        return bsr.get_beam_depth()

    else:
        raise ValueError(study.x_axis)


def get_multi_point_legend_key(study: Study, bsr: BandSizeRatio):
    match study.x_axis:
        case XAxis.band_depth_ratio:
            beam_thickness = bsr.get_beam_depth()
            return f"{beam_thickness:1.0g} mm (Simulation)"

        case _:
            raise ValueError(study.x_axis)


def _get_close_up_subfigure(target_aspect_ratio: float, bsr: BandSizeRatio) -> Image:
    working_dir_end = bsr.run_params.working_dir.parts[-1]
    local_copy_working_dir = plot_data_base / working_dir_end

    images = list(local_copy_working_dir.glob("Case-*.png"))
    if len(images) != 1:
        raise ValueError(images)

    image_fn = images.pop()

    full_image = Image.open(image_fn)

    cropped_image = image_cropper.get_dilation_region(target_aspect_ratio, full_image)

    return cropped_image


def _figure_setup(plot_type: PlotType, study: Study, fn_override: str | None = None):
    """Common stuff for the figures"""
    DPI = 150
    SCALE_DOWN = 1.25  # Was 1.5

    figsize_inches = (
        active_config.screenshot_res.width / SCALE_DOWN / DPI,
        active_config.screenshot_res.height / SCALE_DOWN / DPI,
    )
    figsize_dots = [DPI * i for i in figsize_inches]

    fig, ax = plt.subplots(
        1,
        1,
        sharex=True,
        figsize=figsize_inches,
        dpi=DPI,
    )

    if fn_override:
        fn = fn_override + ".png"

    else:
        fn = f"E4-{study.name}-{plot_type.name}-{_bsr_list_hash(study.band_size_ratios)}.png"

    fig_fn = (
        graph_output_base
        / fn
    )

    return fig, ax, DPI, figsize_dots, fig_fn

def _get_applicable_experimental_results(study: Study):
    keys = set()
    for bsr in study.band_size_ratios:
        keys.add(bsr.run_params.working_dir.name)

    _lookups = {
        "DA": (0, {}),    # 1mm A
        "CU": (1, {"markerfacecolor": 'none'}),    # 1mm B
        "CT": (2, {}),    # 3mm A
        "DO": (3, {"markerfacecolor": 'none'}) ,    # 3mm B
    }

    matched_idxs_args = [v for k, v in _lookups.items() if k in keys]                          
    
    all_res = aletha_csv.read_experimental_data()

    for idx, args in matched_idxs_args:
        yield all_res[idx], args

def make_multiple_plot_data(plot_type: PlotType, study: Study):

    if plot_type.get_data_source() != DataSource.single_band_single_point:
        raise ValueError("This is only for single_band_single_point")

    fig, ax, dpi, figsize_dots, fig_fn = _figure_setup(plot_type, study)

    # Cycle markers for each beam thickness, and also cycle the colours

    cmap_def = matplotlib.cm.get_cmap("tab20")
    c_cycle = cycler(color=cmap_def.colors)
    c_cycle = cycler(color="kkbb")
    m_cycle = cycler(marker=["s", "s", "^", "^"])
    product_cycle = c_cycle + m_cycle
    ax.set_prop_cycle(product_cycle)

    def sort_key(band_size_ratio: BandSizeRatio):
        return get_x_axis_val_raw(study, band_size_ratio)

    # Cycle the marker face colour if there's a duplicate
    raw_legend_count = collections.Counter()

    for bsr in sorted(study.band_size_ratios, key=sort_key):

        beam_thickness = bsr.get_beam_depth()

        x_points, y_points = [], []

        maj_cutoff = abs(bsr.major_band_threshold())

        major_bands = [band for band in bsr.bands if abs(band.band_size) > maj_cutoff]
        for transformation_band in major_bands:

            x_points.append(transformation_band.depth / beam_thickness)
            y_points.append(transformation_band.width / beam_thickness)

        legend_key_raw = get_multi_point_legend_key(study, bsr)
        raw_legend_count[legend_key_raw] += 1
        sim_idx = raw_legend_count[legend_key_raw]
        sim_letter = chr(64 + sim_idx)
        legend_key = f"{legend_key_raw} {sim_letter}"

        kwargs = {"linestyle": "", "label": legend_key, "markersize": 5}
        if sim_idx == 1:
            pass

        elif sim_idx == 2:
            kwargs["markerfacecolor"] = "none"

        else:
            print(f"No more markfacecolor options for {legend_key_raw}")

        ax.plot(x_points, y_points, **kwargs)

        print(bsr.run_params.working_dir.name, legend_key)


    # Experimental results
    exp_to_plot = list(_get_applicable_experimental_results(study))
    for exp, args in exp_to_plot:
        # TODO - up to here
        kwargs = {"linestyle": "", "label": exp.key, "markersize": 5}
        kwargs.update(args)
        ax.plot(exp.x, exp.y, **kwargs)


    if study.x_axis.get_x_range():
        plt.xlim(*study.x_axis.get_x_range())

    if plot_type.get_y_axis_limits():
        plt.ylim(*plot_type.get_y_axis_limits())

    plt.legend()
    plt.savefig(
        fig_fn,
        dpi=2 * dpi,
        bbox_inches="tight",
    )

    print(fig_fn)


def make_aletha_recreate_paper():
    for exp_data in aletha_csv.get_band_exp_data():
        plot_type = PlotType.num_bands
        fig, ax, dpi, figsize_dots, fig_fn = _figure_setup(plot_type, study, fn_override=exp_data.key)

        ax.plot(exp_data.x, exp_data.y)
        # TODO - up to here. Error bars etc


def make_main_plot(plot_type: PlotType, study: Study):

    if plot_type.get_data_source() != DataSource.single_study_single_point:
        raise ValueError("This is only for single_study_single_point")

    TILE_N_X = 3  # waas 3
    TILE_N_Y = 3  # Was 4

    fig, ax, dpi, figsize_dots, fig_fn = _figure_setup(plot_type, study)

    # Subfigure tiles dimensions
    tile_size_dots = [int(figsize_dots[0] / TILE_N_X), int(figsize_dots[1] / TILE_N_Y)]
    tile_aspect_ratio = tile_size_dots[0] / tile_size_dots[1]

    def sort_key(band_size_ratio: BandSizeRatio):
        return get_x_axis_val_raw(study, band_size_ratio)

    plot_type_to_data = collections.defaultdict(list)
    plot_type_to_y_error = collections.defaultdict(list)
    x = []
    annotation_bboxes: typing.List[AnnotationBbox] = []

    for idx, bsr in enumerate(sorted(study.band_size_ratios, key=sort_key)):

        if study.x_axis == XAxis.run_index:
            x_val = idx

        elif study.x_axis in (
            XAxis.beam_depth,
            XAxis.initiation_variation,
            XAxis.initiation_spacing,
            XAxis.dilation_max,
        ):
            x_val = get_x_axis_val_raw(study, bsr)

        else:
            raise ValueError(study.x_axis)

        x.append(x_val)

        if plot_type == PlotType.maj_ratio:
            res = bsr.get_major_band_count_ratio()

        elif plot_type == PlotType.maj_spacing:
            MM_TO_NM = 1_000
            res = MM_TO_NM * bsr.get_major_band_spacing_sd()

        elif plot_type == PlotType.num_bands:
            res = bsr.get_num_maj_bands_full_length()

        else:
            raise ValueError(plot_type)

        plot_type_to_data[plot_type].append(res.val)
        if res.y_sd != None:
            plot_type_to_y_error[plot_type].append(res.y_sd)

        # Annotations?
        working_dir_end = bsr.run_params.working_dir.parts[-1]
        if working_dir_end in study.images_to_annotate:
            cropped_sub_image = _get_close_up_subfigure(
                target_aspect_ratio=tile_aspect_ratio, bsr=bsr
            )

            zoom_x = tile_size_dots[0] / cropped_sub_image.width
            zoom_y = tile_size_dots[1] / cropped_sub_image.height
            buffer = 2.0
            zoom = 0.5 * (zoom_x + zoom_y) * 72.0 / dpi / buffer

            imagebox = OffsetImage(cropped_sub_image, zoom=zoom)
            imagebox.image.axes = ax

            ab = AnnotationBbox(
                imagebox,
                (x_val, res.val),
                xybox=(120.0, -80.0),
                xycoords="data",
                boxcoords="offset points",
                pad=0.5,
                arrowprops=dict(
                    arrowstyle="->",
                    # connectionstyle="angle,angleA=0,angleB=90,rad=3")
                ),
            )

            ax.add_artist(ab)
            annotation_bboxes.append(ab)
    if plot_type_to_y_error[plot_type]:
        yerr = plot_type_to_y_error[plot_type]

    else:
        yerr = None

    if yerr:    
        main_line, caplines, barlinecols = ax.errorbar(x, plot_type_to_data[plot_type], yerr=yerr, marker='.', capsize=4, elinewidth=0.8, label=plot_type.value)

        paths = [main_line.get_path()]
        for cl in caplines:
            paths.append(cl.get_path())

        for blc in barlinecols:
            paths.extend(blc.get_paths())

    else:
        main_line, = ax.plot(x, plot_type_to_data[plot_type], marker='.', label=plot_type.value)
        paths = [main_line.get_path()]


    main_lines = [
        main_line,
    ]

    plt.xlabel(study.x_axis.get_x_label())
    plt.ylabel(plot_type.value)
    # plt.legend()

    if study.x_axis.get_x_range():
        plt.xlim(*study.x_axis.get_x_range())

    if plot_type.get_y_axis_limits():
        plt.ylim(*plot_type.get_y_axis_limits())

    fig.canvas.draw()

    filter_intersecting = False
    proposed_configurations = annotation_tile.generate_proposed_tiles(
        TILE_N_X,
        TILE_N_Y,
        study.tile_position,
        filter_intersecting,
        ax,
        main_lines,
        annotation_bboxes,
    )

    # plt.savefig(fig_fn, dpi=2*dpi, bbox_inches='tight',)

    annotation_tile.save_best_configuration_to(
        ax,
        paths,
        proposed_configurations,
        fig_fn,
    )


def make_cutoff_example(study: Study):
    DPI = 150

    fig, ax = plt.subplots(
        1,
        1,
        sharex=True,
        figsize=(
            active_config.screenshot_res.width / 2 / DPI,
            active_config.screenshot_res.height / 2 / DPI,
        ),
        dpi=DPI,
    )

    with open(r"C:\Users\Tom Wilson\Dropbox\PhD\Papers\Mike-1-Ceramic Bands\2022-v6\inflection3.csv", "a", newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)

        for bsr in study.band_size_ratios:
            band_sizes = bsr._abs_band_sizes()

            x, y = [], []
            for idx, bs in enumerate(band_sizes):
                x.append(idx / (len(band_sizes) - 1))
                y.append(bs / bsr.get_scale())

                writer.writerow([study.name, bsr.run_params.working_dir.name, bsr.get_beam_depth(), idx / (len(band_sizes) - 1),bs / bsr.get_scale(), bs,  ])

            de = bsr.run_params.working_dir.name
            label_value = friendly_str(get_x_axis_val_raw(study, bsr))
            label = f"{study.x_axis.get_legend_label()}={label_value}"
            (base_line,) = plt.plot(x, y, label=label)

            # Add a proposed "cutoff_line"
            bs_prop = bsr.major_band_threshold() / bsr.get_scale()
            ax.plot(
                [-1, 2.0], [bs_prop, bs_prop], color="k", linestyle=":"
            )  # color=base_line.get_color()

            print(
                label,
                bsr.get_major_band_count_ratio(),
                bsr.get_nominal_upper_size() / bsr.get_scale(),
                bs_prop,
            )

    plt.xlim(0.0, 1.0)
    # plt.ylim(1.0, 30.0)

    plt.legend()

    plt.xlabel("Band size rank")
    plt.ylabel("Band size")

    fig_fn = (
        graph_output_base
        / f"E5-{study.name}-cutoff_demo-{_bsr_list_hash(study.band_size_ratios)}.png"
    )
    plt.savefig(fig_fn, dpi=2 * DPI)

    # plt.clear()


def print_band_size_info(study: Study):
    # TODO - refactor to use "Study" everywhere
    csv_fn = (
        graph_output_base
        / f"{study.name}-cutoff_demo-{_bsr_list_hash(study.band_size_ratios)}.csv"
    )
    with open(csv_fn, "w", newline="") as csv_file:
        f_out = csv.writer(csv_file)

        for bsr in study.band_size_ratios:
            for maj_ratio, band in bsr.get_band_and_maj_ratio():
                bits = [
                    bsr.run_params.working_dir.name,
                    band.x,
                    abs(band.band_size),
                    abs(band.band_size) / bsr.get_scale(),
                    maj_ratio,
                ]
                f_out.writerow(bits)


def run_study(study: Study):
    for plot_type in study.x_axis.get_applicable_plot_types():

        data_source = plot_type.get_data_source()
        match data_source:
            case DataSource.single_study_single_point:
                make_main_plot(plot_type, study)

            case DataSource.single_band_single_point:
                make_multiple_plot_data(plot_type, study)

            case _:
                raise ValueError(data_source)

    print_band_size_info(study)

    make_cutoff_example(study)


if __name__ == "__main__":

    TP = annotation_tile.TilePosition

    # cherry_pick = list(generate_plot_data_specified(["CM", "CO", "CT"]))
    # TODO - include the x-range, y-range, etc in these.
    studies = [

        # generate_plot_data_specified("AspectCompareSub2", XAxis.band_depth_ratio, ["DA",], tile_position=TP.top | TP.bottom, images_to_annotate={},),

        # generate_plot_data_specified("AspectCompareSub", XAxis.band_depth_ratio, ["DA", "CU", "CO", "CP", "CQ", "CR", "CS", "CT"], tile_position=TP.top | TP.bottom, images_to_annotate={},),
        
        # generate_plot_data_specified("AspectComparePaper", XAxis.band_depth_ratio, ["DA", "CT"], tile_position=TP.top | TP.bottom, images_to_annotate={},),
        generate_plot_data_specified("AspectComparePaper1mm", XAxis.band_depth_ratio, ["DA", "CU"], tile_position=TP.top | TP.bottom, images_to_annotate={},),
        generate_plot_data_specified("AspectComparePaper3mm", XAxis.band_depth_ratio, ["CT", "DO"], tile_position=TP.top | TP.bottom, images_to_annotate={},),



        # generate_plot_data_range("AspectCompareAll", XAxis.band_depth_ratio, "CM", "DR", tile_position=TP.top | TP.bottom, images_to_annotate={},),


        # generate_plot_data_specified("SpacingVariation4", XAxis.initiation_spacing, ["C3", "CI", "CK", "CJ", "CL", "F8", "F9", "FA", "FB", "FC", "FD", "FE"], tile_position=TP.top | TP.bottom, images_to_annotate={"CJ", "CL", "F8", "FA", "FB", "FC", }, accept_not_the_last_increment=True),
        # generate_plot_data_specified("InitationVariation", XAxis.initiation_variation, ["C3", "CF", "CG", "CH"], tile_position=TP.top | TP.bottom, images_to_annotate={"C3", "CF", "CG", "CH",}),
        # generate_plot_data_range("SpreadStudy", XAxis.run_index, "DZ", "E5", tile_position=TP.top, images_to_annotate={"DZ", "E1", "E5",}),

        # generate_plot_data_range( "BeamDepth3", XAxis.beam_depth, "CM", "DR", tile_position=TP.edges, images_to_annotate={"CM", "CO", "CQ", "CT",}),
        # generate_plot_data_specified("CherryPick2", XAxis.beam_depth, ["CM", "CO",], tile_position=TP.top, images_to_annotate={"CM", "CO",})
    ]
    # generate_plot_data_range("ELocalMax", XAxis.dilation_max, "C4", "C9", tile_position=TP.top, images_to_annotate={"C4", "C6", "C9",}),
    for study in studies:
        run_study(study)
        # make_cutoff_example(study)

    exit()
    dir_ends = ["CM", "CN", "CO", "CP", "CQ", "CR", "CS", "CT"]

    for de in dir_ends:
        d = os.path.join("", de)

        comp = make_band_min_maj_comparison(d)

        band_sizes = comp._abs_band_sizes()

        x, y = [], []
        for idx, bs in enumerate(band_sizes):
            x.append(idx / (len(band_sizes) - 1))
            y.append(bs / comp.get_scale())

        label = f"{de} ScaleY={comp.run_params.scale_model_y}"

        (base_line,) = plt.plot(x, y, label=label)

        # Add a proposed "cutoff_line"
        bs_prop = comp.major_band_threshold() / comp.get_scale()
        plt.plot(
            [-1, 2.0], [bs_prop, bs_prop], color=base_line.get_color(), linestyle="--"
        )

        print(
            label,
            comp.get_major_band_count_ratio(),
            comp.get_nominal_upper_size() / comp.get_scale(),
            bs_prop,
        )

    plt.xlim(0.0, 1.0)
    plt.ylim(1.0, 30.0)

    plt.legend()
    plt.show()
