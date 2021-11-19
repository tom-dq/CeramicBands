"""Utility for tiling subfigures in a matplotlib figure in the "least bad" way."""

import itertools
import math
import typing

from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox


class ProposedTile(typing.NamedTuple):
    i_x: int
    i_y: int
    annotation_bbox: AnnotationBbox


class ProposedConfiguration(typing.NamedTuple):
    n_x: int
    n_y: int
    ax: Axes
    tiles: typing.List[ProposedTile]


def generate_proposed_tiles(n_x: int, n_y: int, ax: Axes, annotation_bboxes: typing.List[AnnotationBbox]) -> typing.Iterable[ProposedConfiguration]:
    

    all_tiles = list(itertools.product(range(n_x), range(n_y)))

    # Preflight check - how many options are we dealing with?
    n_perms = math.perm(len(all_tiles), len(annotation_bboxes))
    print(f"Total options on the table: {n_perms}")

    for tile_proposal in itertools.permutations(all_tiles, len(annotation_bboxes)):
        tiles = []
        for (i_x, i_y), annotation_bbox in itertools.zip_longest(tile_proposal, annotation_bboxes):
            proposed_tile = ProposedTile(i_x=i_x, i_y=i_y, annotation_bbox=annotation_bbox)
            tiles.append(proposed_tile)

        yield ProposedConfiguration(
            n_x=n_x,
            n_y=n_y,
            ax=ax,
            tiles=tiles,
        )



if __name__ == "__main__":
    for x in generate_proposed_tiles(3, 2, None, ["A", "B"]):
        print(x)