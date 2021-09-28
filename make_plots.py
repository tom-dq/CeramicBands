import pathlib
import statistics
import typing
import os

import scipy.stats

import matplotlib.pyplot as plt
import matplotlib.ticker

from tables import Table
import common_types

import main
import history

# To get pickle to unpickle
from main import CheckpointState, save_state
from main import RunParams
from main import ModelFreedomCase
from main import Ratchet
from main import PrestrainUpdate
from main import ResultFrame


class BandSizeRatio(typing.NamedTuple):
    run_params: RunParams
    bands: typing.List[history.TransformationBand]


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
        return self.get_nominal_lower_size()

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
    last_minor_inc = max(minor_incs)

    one_case = [c for c in cases_subset if c.minor_inc == last_minor_inc]

    if len(one_case) != 1:
        raise ValueError(one_case)

    return one_case.pop()


def make_band_min_maj_comparison(working_dir: typing.Union[pathlib.Path, str]) -> BandSizeRatio:

    working_dir = pathlib.Path(working_dir)
    saved_state = main.load_state(working_dir)

    with history.DB(working_dir / "history.db") as db:
        cases = list(db.get_all(history.ResultCase))
        last_case = _get_last_result_case_num(saved_state, cases)

        band_skeleton = history.TransformationBand._all_nones()._replace(result_case_num=last_case.num)

        last_case_bands = list(db.get_all_matching(band_skeleton))


    return BandSizeRatio(run_params=saved_state.run_params, bands=last_case_bands)


if __name__ == "__main__":

    dir_ends = ['CM', 'CN', 'CO', 'CP', 'CQ', 'CR', 'CS', 'CT']


    for de in dir_ends:
        d = os.path.join(r"C:\Users\Tom Wilson\Dropbox\PhD\Ceramic Bands Source Models\Test Output", de)

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
    plt.ylim(1.0, 50.0)

    plt.legend()
    plt.show()
