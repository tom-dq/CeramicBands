"""Utility for tiling subfigures in a matplotlib figure in the "least bad" way."""

import typing

from matplotlib.axes import Axes


class ProposedTile(typing.NamedTuple):
    i_x: int
    i_y: int
    n_x: int
    n_y: int
    ax: Axes


def generate_propotiles(ax: Axes) -> typing.Iterable[ProposedTile]:
    pass