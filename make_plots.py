import pathlib
import typing

import matplotlib.pyplot as plt
import matplotlib.ticker

from tables import Table
import common_types

import main
import history


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


def make_band_min_maj_comparison(working_dir: typing.Union[pathlib.Path, str]):

    working_dir = pathlib.Path(working_dir)
    saved_state = main.load_state(working_dir)

    with history.DB(working_dir / "history.db") as db:
        cases = list(db.get_all(history.ResultCase))
        bands = list(db.get_all(history.TransformationBand))

    for c in cases:
        print(c)

    for b in bands:
        print(b)



if __name__ == "__main__":
    make_band_min_maj_comparison(r"C:\Users\Tom Wilson\Dropbox\PhD\Ceramic Bands Source Models\Test Output\CM")
