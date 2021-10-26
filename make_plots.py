import pathlib
import statistics
import typing
import os
import enum
import itertools
import collections
import hashlib
import csv
from networkx.drawing import layout
from networkx.readwrite import graph6

import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.markers

from tables import Table
import common_types

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


def _generate_plot_data(gen_relevant_subdirectories) -> typing.Iterable[BandSizeRatio]:
    # Get the relevant subdirectories for inclusion

    for working_dir in gen_relevant_subdirectories():
        try:
            band_size_ratio = make_band_min_maj_comparison(working_dir)
            if band_size_ratio.result_case_num > 1000:
                yield band_size_ratio

        except NoResultException:
            pass


def generate_plot_data_range(first_considered_subdir: str, last_considered_subdir: str) -> typing.Iterable[BandSizeRatio]:
    
    def gen_relevant_subdirectories():
        min_hex = int(first_considered_subdir, base=36)
        max_hex = int(last_considered_subdir, base=36)

        for working_dir in plot_data_base.iterdir():
            if working_dir.is_dir():
                this_hex = int(working_dir.parts[-1], base=36)
                if min_hex <= this_hex <= max_hex:
                    yield working_dir

    yield from _generate_plot_data(gen_relevant_subdirectories)


def generate_plot_data_specified(dir_ends: typing.List[str]) -> typing.Iterable[BandSizeRatio]:
    def gen_relevant_subdirectories():
        for de in dir_ends:
            yield plot_data_base / de

    yield from _generate_plot_data(gen_relevant_subdirectories)


class PlotType(enum.Enum):
    maj_ratio = "Proportion of major transformation bands"
    maj_spacing = "Average spacing between major bands"


def _fig_fn(plot_type: typing.Optional[PlotType]=None):
    name_end = plot_type.name if plot_type else "depth_compare"
    return graph_output_base / f"{name_end}.png"


def _bsr_list_hash(bsr_list: typing.List[BandSizeRatio]) -> str:
    """Unique string for the output plots so they don't overwrite each other."""

    dir_ends = [bsr.run_params.working_dir.name for bsr in bsr_list]
    hash_bit = hashlib.md5('-'.join(dir_ends).encode()).hexdigest()[0:6]

    return f"{len(bsr_list)}-{hash_bit}"


def make_main_plot(band_size_ratios: typing.List[BandSizeRatio]):
    
    DPI = 200

    def sort_key(band_size_ratio: BandSizeRatio):
        return band_size_ratio.run_params.scale_model_y

    fig, ax = plt.subplots(1, 1, sharex=True, 
        figsize=(active_config.screenshot_res.width/2/DPI, active_config.screenshot_res.height/2/DPI), 
        dpi=DPI,
    )

    plot_type_to_data = collections.defaultdict(list)
    x = []
    for bsr in sorted(band_size_ratios, key=sort_key):
        x.append(bsr.run_params.scale_model_y)

        for plot_type in PlotType:
            if plot_type == PlotType.maj_ratio:
                val = bsr.get_major_band_count_ratio()
            
            elif plot_type == PlotType.maj_spacing:
                val = bsr.get_major_band_spacing()

            else:
                raise ValueError(plot_type)

            plot_type_to_data[plot_type].append(val)

    for plot_type in PlotType:
        ax.plot(x, plot_type_to_data[plot_type], marker='.', label=plot_type.value)

    plt.xlabel("Relative depth of beam")
    plt.legend()    


    fig_fn = graph_output_base / f"depth_compare-{_bsr_list_hash(band_size_ratios)}.png"
    plt.savefig(fig_fn, dpi=DPI, layout='tight',)

    plt.show()


def make_cutoff_example(band_size_ratios: typing.List[BandSizeRatio]):
    DPI = 150

    fig, ax = plt.subplots(1, 1, sharex=True, 
        figsize=(active_config.screenshot_res.width/2/DPI, active_config.screenshot_res.height/2/DPI), 
        dpi=DPI,
    )

    for bsr in band_size_ratios:
        band_sizes = bsr._abs_band_sizes()

        x, y = [], []
        for idx, bs in enumerate(band_sizes):
            x.append(idx/(len(band_sizes)-1))
            y.append(bs / bsr.get_scale())

        de = bsr.run_params.working_dir.name
        label = f"ScaleY={bsr.run_params.scale_model_y}"

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

    fig_fn = graph_output_base / f"cutoff_demo-{_bsr_list_hash(band_size_ratios)}.png"
    plt.savefig(fig_fn, dpi=2*DPI)

    plt.show()


def print_band_size_info(band_size_ratios: typing.List[BandSizeRatio]):
    
    csv_fn = graph_output_base / f"cutoff_demo-{_bsr_list_hash(band_size_ratios)}.csv"
    with open(csv_fn, 'w', newline='') as csv_file:
        f_out = csv.writer(csv_file)

        for bsr in band_size_ratios:
            for maj_ratio, band in bsr.get_band_and_maj_ratio():
                bits = [bsr.run_params.working_dir.name, band.x, abs(band.band_size), abs(band.band_size) / bsr.get_scale(), maj_ratio]
                f_out.writerow(bits)

        
        



_ref_data_ = """
CM ScaleY=0.3 0.6122448979591837 13.551480668778519 2.5689350835973146
CN ScaleY=0.4 0.42857142857142855 31.245934998141536 4.780741874767692
CO ScaleY=0.5 0.42857142857142855 43.65032351469947 6.331290439337435
CP ScaleY=0.6 0.3877551020408163 60.85611233055029 8.482014041318784
CQ ScaleY=0.7 0.30612244897959184 87.69479570513748 11.836849463142185
CR ScaleY=0.8 0.30612244897959184 125.7815875503112 16.5976984437889
CS ScaleY=0.9 0.2708333333333333 164.4629790006403 21.432872375080034
CT ScaleY=1.0 0.2708333333333333 210.52437503402243 27.190546879252803
CU ScaleY=0.35 0.5306122448979592 19.326962734418274 3.2908703418022847
CV ScaleY=0.45 0.42857142857142855 36.808001295100496 5.476000161887561
CW ScaleY=0.55 0.3877551020408163 67.40343924177411 9.300429905221764
CX ScaleY=0.65 0.32653061224489793 94.41709670331198 12.677137087913996
CY ScaleY=0.75 0.30612244897959184 98.27476475028878 13.159345593786098
CZ ScaleY=0.85 0.3125 138.393837227894 18.174229653486748
D0 ScaleY=0.95 0.2857142857142857 147.13211317114133 19.266514146392666
D2 ScaleY=0.375 0.5102040816326531 23.693954746238813 3.836744343279852
D5 ScaleY=0.525 0.40816326530612246 53.46577949656183 7.558222437070229
D9 ScaleY=0.475 0.42857142857142855 39.15918950773319 5.7698986884666486
DA ScaleY=0.325 0.5918367346938775 16.737657708939533 2.9672072136174417
DB ScaleY=0.425 0.46938775510204084 19.423579449551845 3.30294743119398
DH ScaleY=0.625 0.3673469387755102 81.87139205210727 11.108924006513409
DI ScaleY=0.675 0.3469387755102041 70.16913119578544 9.64614139947318
DJ ScaleY=0.725 0.3469387755102041 98.03954773511492 13.129943466889365
DL ScaleY=0.825 0.2708333333333333 137.38533046043057 18.04816630755382
DN ScaleY=0.925 0.2653061224489796 204.31144367867878 26.413930459834845
DO ScaleY=0.975 0.2765957446808511 175.28410456142828 22.785513070178535
DP ScaleY=0.575 0.38 70.30812317445681 9.663515396807101
DQ ScaleY=0.775 0.32653061224489793 100.7993644043031 13.474920550537888
DR ScaleY=0.875 0.30612244897959184 131.33836092919856 17.29229511614982
"""

if __name__ == "__main__":

    
    # cherry_pick = list(generate_plot_data_specified(["CM", "CO", "CT"]))
    cherry_pick = list(generate_plot_data_specified(["CM"]))

    print_band_size_info(cherry_pick)

    make_cutoff_example(cherry_pick)

    exit()

    all_band_sizes = list(generate_plot_data_range("CM", "DR"))
    print_band_size_info(all_band_sizes)

    make_cutoff_example(all_band_sizes)
    # make_main_plot(all_band_sizes)

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
