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
from matplotlib.patches import Polygon
from networkx.drawing import layout
from networkx.readwrite import graph6

import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.markers

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from PIL import Image

from tables import Table
import common_types
import image_cropper

from config import active_config

import main
import history

# To get pickle to unpickle
from main import CheckpointState, save_state
from main import RunParams
from main import ModelFreedomCase
from main import Ratchet
from main import PrestrainUpdate
from main import ResultFrame

T_Path = typing.Union[pathlib.Path, str]

# plot_data_base=pathlib.Path(r"C:\Users\Tom Wilson\Dropbox\PhD\Ceramic Bands Source Models\Test Output")
plot_data_base=pathlib.Path(r"C:\Users\Tom Wilson\Documents\CeramicBandData\outputs\192.168.1.109+8080\v7\pics")
graph_output_base = pathlib.Path(r"C:\Users\Tom Wilson\Dropbox\PhD\Papers\Mike-1-Ceramic Bands\2021-v5")

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
        scaled = [b/min_abs_size for b in band_sizes]
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

        bs_prop = bs_min + 1/8 * bs_diff
        return bs_prop

    def get_major_band_count_ratio(self) -> float:
        """Proportion of the transformation bands which are "major" bands."""

        cutoff = self.major_band_threshold()

        maj_bands = [bs for bs in self._abs_band_sizes() if bs > cutoff]
        return len(maj_bands) / len(self.bands)

    def get_major_band_spacing(self) -> float:
        """Average distance between major bands."""

        cutoff = self.major_band_threshold()

        maj_bands = [b for b in self.bands if abs(b.band_size) > cutoff]

        maj_band_x_vals = sorted(bs.x for bs in maj_bands)

        n_band_gaps = len(maj_bands) - 1
        maj_band_span = maj_band_x_vals[-1] - maj_band_x_vals[0]

        return maj_band_span / n_band_gaps

    def get_band_and_maj_ratio(self) -> typing.Iterable[typing.Tuple[float, history.TransformationBand]]:
        """Returns the bands in order, with the ratio of the band size. So if larger than one, it's a major band."""

        cutoff = self.major_band_threshold()

        for band in sorted(self.bands):
            yield abs(band.band_size) / cutoff, band

    def get_num_maj_bands_full_length(self) -> float:
        """Assume the major band density seen in the current range is replicated along the full length"""

        maj_band_spacing = self.get_major_band_spacing()
        return SPECIMEN_NOMINAL_LENGTH_MM / maj_band_spacing

    def get_beam_depth(self) -> float:

        return FULL_BEAM_HEIGHT * self.run_params.scale_model_y


def make_example_table() -> Table:
    STRESS_START = 400
    stress_end = 450
    dilation_ratio = 0.008

    prestrain_table = Table([
        common_types.XY(0.0, 0.0),
        common_types.XY(STRESS_START, 0.0),
        common_types.XY(stress_end, -1*dilation_ratio),
        common_types.XY(stress_end+200, -1*dilation_ratio),
    ])

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


def _get_last_result_case_num(saved_state: CheckpointState, cases: typing.List[history.ResultCase]) -> history.ResultCase:
    last_maj_inc = saved_state.run_params.get_maj_incs(one_based=True)[-1]

    if any(c.major_inc > last_maj_inc for c in cases):
        raise ValueError(f"Logical error - only expected to have up to case {last_maj_inc}")

    cases_subset = [c for c in cases if c.major_inc == last_maj_inc]
    minor_incs = [c.minor_inc for c in cases_subset]

    if not minor_incs:
        raise NoResultException(minor_incs)

    last_minor_inc = max(minor_incs)

    one_case = [c for c in cases_subset if c.minor_inc == last_minor_inc]

    if len(one_case) != 1:
        raise NoResultException(one_case)

    return one_case.pop()


@functools.lru_cache(maxsize=256)
def make_band_min_maj_comparison(working_dir: T_Path) -> BandSizeRatio:

    working_dir = pathlib.Path(working_dir)
    try:
        saved_state = main.load_state(working_dir)
    except FileNotFoundError:

        raise NoResultException()


    with history.DB(working_dir / "history.db") as db:
        cases = list(db.get_all(history.ResultCase))
        last_case = _get_last_result_case_num(saved_state, cases)

        print(working_dir.parts[-1], last_case._replace(name=''), saved_state.run_params.scale_model_y)

        band_skeleton = history.TransformationBand._all_nones()._replace(result_case_num=last_case.num)

        last_case_bands = list(db.get_all_matching(band_skeleton))


    return BandSizeRatio(run_params=saved_state.run_params, bands=last_case_bands, result_case_num=last_case.num)


class PlotType(enum.Enum):
    maj_ratio = "Proportion of major transformation bands"
    maj_spacing = "Average spacing between major bands"
    num_bands = f"Number of bands over {SPECIMEN_NOMINAL_LENGTH_MM} mm length"


class XAxis(enum.Enum):
    beam_depth = enum.auto()
    dilation_max = enum.auto()
    run_index = enum.auto()  # Just order the simulations one by one.
    initiation_variation = enum.auto()
    initiation_spacing = enum.auto()

    def get_x_label(self) -> str:
        d = {
            XAxis.beam_depth: "Beam Depth (mm)",
            XAxis.dilation_max: "Dilation (Max)",
            XAxis.run_index: "Run Index",
            XAxis.initiation_variation: "Initiation Variation",
            XAxis.initiation_spacing: "Initiation Spacing",
        }
        
        return d[self]

    def get_x_range(self) -> typing.Optional[typing.Tuple[float, float]]:
        if self == XAxis.beam_depth:
            return (0.0, 3.5)

        return None

    def get_legend_label(self) -> str:
        d = {
            XAxis.beam_depth: "BeamDepth",
            XAxis.dilation_max: "DilationRatio",
            XAxis.run_index: "Run",
            XAxis.initiation_variation: "InitiationVariation",
            XAxis.initiation_spacing: "InitiationSpacing",
        }

        return d[self]

class Study(typing.NamedTuple):
    name: str
    band_size_ratios: typing.List[BandSizeRatio]
    x_axis: XAxis
    images_to_annotate: typing.Set[str]


def _generate_study_data(name, x_axis, images_to_annotate: typing.Optional[typing.Set[str]], gen_relevant_subdirectories) -> Study:
    # Get the relevant subdirectories for inclusion

    if images_to_annotate is None:
        images_to_annotate = set()


    def make_bsrs():
        for working_dir in gen_relevant_subdirectories():
            try:
                band_size_ratio = make_band_min_maj_comparison(working_dir)
                if band_size_ratio.result_case_num > 800:
                    yield band_size_ratio

            except NoResultException:
                pass

    return Study(name=name, band_size_ratios=list(make_bsrs()), x_axis=x_axis, images_to_annotate=images_to_annotate)


def generate_plot_data_range(name, x_axis, first_considered_subdir: str, last_considered_subdir: str, images_to_annotate: typing.Optional[typing.Set[str]] = None) -> Study:
    
    def gen_relevant_subdirectories():
        min_hex = int(first_considered_subdir, base=36)
        max_hex = int(last_considered_subdir, base=36)

        for working_dir in plot_data_base.iterdir():
            if working_dir.is_dir():
                this_hex = int(working_dir.parts[-1], base=36)
                if min_hex <= this_hex <= max_hex:
                    yield working_dir

    return _generate_study_data(name, x_axis, images_to_annotate, gen_relevant_subdirectories)


def generate_plot_data_specified(name, x_axis, dir_ends: typing.List[str], images_to_annotate: typing.Optional[typing.Set[str]] = None) -> Study:

    def gen_relevant_subdirectories():
        for de in dir_ends:
            yield plot_data_base / de

    return _generate_study_data(name, x_axis, images_to_annotate, gen_relevant_subdirectories)






def _fig_fn(plot_type: typing.Optional[PlotType]=None):
    name_end = plot_type.name if plot_type else "depth_compare"
    return graph_output_base / f"{name_end}.png"


def _bsr_list_hash(bsr_list: typing.List[BandSizeRatio]) -> str:
    """Unique string for the output plots so they don't overwrite each other."""

    dir_ends = [bsr.run_params.working_dir.name for bsr in bsr_list]
    hash_bit = hashlib.md5('-'.join(dir_ends).encode()).hexdigest()[0:6]

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

    else:
        raise ValueError(study.x_axis)


def _get_close_up_subfigure(bsr: BandSizeRatio) -> Image:
    working_dir_end = bsr.run_params.working_dir.parts[-1]
    local_copy_working_dir = plot_data_base / working_dir_end

    images = list(local_copy_working_dir.glob("Case-*.png"))
    if len(images) != 1:
        raise ValueError(images)

    image_fn = images.pop()

    full_image = Image.open(image_fn)

    cropped_image = image_cropper.get_dilation_region(1.2, full_image)

    return cropped_image




def make_main_plot(plot_type: PlotType, study: Study):
    
    DPI = 150


    def sort_key(band_size_ratio: BandSizeRatio):
        return band_size_ratio.run_params.scale_model_y

    fig, ax = plt.subplots(1, 1, sharex=True, 
        figsize=(active_config.screenshot_res.height/2/DPI, active_config.screenshot_res.height/2/DPI), 
        dpi=DPI,
    )

    plot_type_to_data = collections.defaultdict(list)
    x = []
    annotation_bboxes: typing.List[AnnotationBbox] = []

    for idx, bsr in enumerate(sorted(study.band_size_ratios, key=sort_key)):

        if study.x_axis == XAxis.run_index:
            x_val = idx

        elif study.x_axis in (XAxis.beam_depth, XAxis.initiation_variation, XAxis.initiation_spacing, XAxis.dilation_max):
            x_val = get_x_axis_val_raw(study, bsr)
            

        else:
            raise ValueError(study.x_axis)

        x.append(x_val)

        if plot_type == PlotType.maj_ratio:
            y_val = bsr.get_major_band_count_ratio()
        
        elif plot_type == PlotType.maj_spacing:
            MM_TO_NM = 1_000
            y_val = MM_TO_NM * bsr.get_major_band_spacing()

        elif plot_type == PlotType.num_bands:
            y_val = bsr.get_num_maj_bands_full_length()

        else:
            raise ValueError(plot_type)

        plot_type_to_data[plot_type].append(y_val)

        # Annotations?
        working_dir_end = bsr.run_params.working_dir.parts[-1]
        if working_dir_end in study.images_to_annotate:
            cropped_sub_image = _get_close_up_subfigure(bsr)
            imagebox = OffsetImage(cropped_sub_image, zoom=0.2)
            imagebox.image.axes = ax

            ab = AnnotationBbox(imagebox, (x_val, y_val),
                    xybox=(120., -80.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5,
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=3")
                    )

            ax.add_artist(ab)
            annotation_bboxes.append(ab)

    ax.plot(x, plot_type_to_data[plot_type], marker='.', label=plot_type.value)

    plt.xlabel(study.x_axis.get_x_label())
    plt.ylabel(plot_type.value)
    # plt.legend()    

    if study.x_axis.get_x_range():
        plt.xlim(*study.x_axis.get_x_range())


    if plot_type == PlotType.num_bands:
        # plt.ylim(0, 150)
        pass

    elif plot_type == PlotType.maj_spacing:
        # plt.ylim(0, 1000)
        pass
    
    elif plot_type == PlotType.maj_ratio:
        pass

    else:
        raise ValueError(plot_type)


    fig_fn = graph_output_base / f"{study.name}-{plot_type.name}-{_bsr_list_hash(study.band_size_ratios)}.png"
    plt.savefig(fig_fn, dpi=2*DPI, bbox_inches='tight',)

    # plt.show()


def make_cutoff_example(study: Study):
    DPI = 150

    fig, ax = plt.subplots(1, 1, sharex=True, 
        figsize=(active_config.screenshot_res.width/2/DPI, active_config.screenshot_res.height/2/DPI), 
        dpi=DPI,
    )

    for bsr in study.band_size_ratios:
        band_sizes = bsr._abs_band_sizes()

        x, y = [], []
        for idx, bs in enumerate(band_sizes):
            x.append(idx/(len(band_sizes)-1))
            y.append(bs / bsr.get_scale())

        de = bsr.run_params.working_dir.name
        label_value = friendly_str(get_x_axis_val_raw(study, bsr))
        label = f"{study.x_axis.get_legend_label()}={label_value}"
        base_line, = plt.plot(x, y, label=label)

        # Add a proposed "cutoff_line"
        bs_prop = bsr.major_band_threshold() / bsr.get_scale()
        ax.plot([-1, 2.0], [bs_prop, bs_prop], color='k', linestyle=':')  # color=base_line.get_color()

        print(label, bsr.get_major_band_count_ratio(), bsr.get_nominal_upper_size() / bsr.get_scale(), bs_prop)

    plt.xlim(0.0, 1.0)
    # plt.ylim(1.0, 30.0)

    plt.legend()

    plt.xlabel("Band size rank")
    plt.ylabel("Band size")

    fig_fn = graph_output_base / f"{study.name}-cutoff_demo-{_bsr_list_hash(study.band_size_ratios)}.png"
    plt.savefig(fig_fn, dpi=2*DPI)

    # plt.clear()


def print_band_size_info(study: Study):
    # TODO - refactor to use "Study" everywhere
    csv_fn = graph_output_base / f"{study.name}-cutoff_demo-{_bsr_list_hash(study.band_size_ratios)}.csv"
    with open(csv_fn, 'w', newline='') as csv_file:
        f_out = csv.writer(csv_file)

        for bsr in study.band_size_ratios:
            for maj_ratio, band in bsr.get_band_and_maj_ratio():
                bits = [bsr.run_params.working_dir.name, band.x, abs(band.band_size), abs(band.band_size) / bsr.get_scale(), maj_ratio]
                f_out.writerow(bits)


def run_study(study: Study):
    for plot_type in PlotType:
        make_main_plot(plot_type, study)

    print_band_size_info(study)

    make_cutoff_example(study)


if __name__ == "__main__":

    
    # cherry_pick = list(generate_plot_data_specified(["CM", "CO", "CT"]))
    studies = [
        # generate_plot_data_range("SpacingVariation", XAxis.initiation_spacing, "CI", "CL"),
        # generate_plot_data_specified("InitationVariation", XAxis.initiation_variation, ["C3", "CF", "CG", "CH"]),
        # generate_plot_data_range("SpreadStudy", XAxis.run_index, "CA", "CE"),
        # generate_plot_data_range("ELocalMax", XAxis.dilation_max, "C4", "C9"),
        # generate_plot_data_range( "BeamDepth", XAxis.beam_depth, "CM", "DR"),
        generate_plot_data_specified("CherryPick", XAxis.beam_depth, ["CM", "CO",], images_to_annotate={"CM",})
    ]

    for study in studies:
        run_study(study)

    exit()
    dir_ends = ['CM', 'CN', 'CO', 'CP', 'CQ', 'CR', 'CS', 'CT']


    for de in dir_ends:
        d = os.path.join('', de)

        comp = make_band_min_maj_comparison(d)

        band_sizes = comp._abs_band_sizes()

        x, y = [], []
        for idx, bs in enumerate(band_sizes):
            x.append(idx/(len(band_sizes)-1))
            y.append(bs / comp.get_scale())

        label = f"{de} ScaleY={comp.run_params.scale_model_y}"

        base_line, = plt.plot(x, y, label=label)

        # Add a proposed "cutoff_line"
        bs_prop = comp.major_band_threshold() / comp.get_scale()
        plt.plot([-1, 2.0], [bs_prop, bs_prop], color=base_line.get_color(), linestyle='--')

        print(label, comp.get_major_band_count_ratio(), comp.get_nominal_upper_size() / comp.get_scale(), bs_prop)

    plt.xlim(0.0, 1.0)
    plt.ylim(1.0, 30.0)

    plt.legend()
    plt.show()
