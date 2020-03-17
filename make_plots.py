import matplotlib.pyplot as plt
import matplotlib.ticker

from tables import Table
from common_types import XY


def make_example_table() -> Table:
    STRESS_START = 400
    stress_end = 450
    dilation_ratio = 0.008

    prestrain_table = Table([
        XY(0.0, 0.0),
        XY(STRESS_START, 0.0),
        XY(stress_end, -1*dilation_ratio),
        XY(stress_end+200, -1*dilation_ratio),
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


if __name__ == "__main__":
    table = make_example_table()
    show_table(table)