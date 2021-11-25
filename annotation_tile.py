"""Utility for tiling subfigures in a matplotlib figure in the "least bad" way."""

import itertools
import math
import typing
import pathlib


import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.markers
import matplotlib.lines

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from PIL import Image

from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox

import image_cropper


class ProposedTile(typing.NamedTuple):
    i_x: int
    i_y: int
    annotation_bbox: AnnotationBbox


class ProposedConfiguration(typing.NamedTuple):
    n_x: int
    n_y: int
    ax: Axes
    tiles: typing.List[ProposedTile]

    def get_axis_fraction_limits(one_tile) -> ...:
        # TODO - return bounding box in 0..1 fraction limits
        pass

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


def apply_tile_configuration(proposed_configuration: ProposedConfiguration):
    
    for proposed_tile in proposed_configuration.tiles:

        # Size and place the figure correctly
        new_xybox = (
            (0.5 + proposed_tile.i_x) / proposed_configuration.n_x,
            (0.5 + proposed_tile.i_y) / proposed_configuration.n_y,
        )

        proposed_tile.annotation_bbox.xybox = new_xybox
        proposed_tile.annotation_bbox.boxcoords = 'axes fraction'

        print(proposed_tile)




def _test_close_up_subfigure(target_aspect_ratio: float, working_dir_end) -> Image:

    plot_data_base=pathlib.Path(r"C:\Users\Tom Wilson\Documents\CeramicBandData\outputs\192.168.1.109+8080\v7\pics")
    
    local_copy_working_dir = plot_data_base / working_dir_end

    images = list(local_copy_working_dir.glob("Case-*.png"))
    if len(images) != 1:
        raise ValueError(images)

    image_fn = images.pop()

    full_image = Image.open(image_fn)

    cropped_image = image_cropper.get_dilation_region(target_aspect_ratio, full_image)

    return cropped_image


def measure_configuration_badness(
        main_line: matplotlib.lines.Line2D,
        proposed_configuration: ProposedConfiguration,
        r,
    ) -> typing.Tuple[int, float]:

    line_path = main_line.get_path()
    inv_trans = proposed_configuration.ax.transData.inverted()
    intersects = 0
    for proposed_tile in proposed_configuration.tiles:
        offset_box_try1 = proposed_tile.annotation_bbox.offsetbox
        offset_box_try2 = proposed_tile.annotation_bbox.clipbox
        exts = inv_trans.transform_bbox(offset_box_try2)

        line_intersects = line_path.intersects_bbox(exts)
        if line_intersects:
            intersects += 1

        print(proposed_tile, line_intersects)

    return intersects, 0.0
    

def make_test_plot():
    screen_h = 2160
    DPI = 150

    TILE_N_X, TILE_N_Y = 3, 3

    figsize_inches=(screen_h/2/DPI, screen_h/2/DPI)
    figsize_dots = [DPI*i for i in figsize_inches]

    # Subfigure tiles dimensions
    tile_size_dots = [int(figsize_dots[0] / TILE_N_X), int(figsize_dots[1] / TILE_N_Y)]
    tile_aspect_ratio = tile_size_dots[0] / tile_size_dots[1]

    fig, ax = plt.subplots(1, 1, sharex=True, 
        figsize=figsize_inches, 
        dpi=DPI,
    )

    main_line, = ax.plot([3,4,5,6], [1.1, 1.2, 1.5, 2.5], marker='.', label="Legend!")

    annotation_bboxes=[]
    for working_dir_end in ["CO", "CM", "CN",]:
        cropped_sub_image = _test_close_up_subfigure(tile_aspect_ratio, working_dir_end)

        zoom_x = tile_size_dots[0] / cropped_sub_image.width
        zoom_y = tile_size_dots[1] / cropped_sub_image.height
        # zoom = 0.8 * 0.5 * (zoom_x + zoom_y) / 2
        buffer = 2.0
        zoom = 0.5 * (zoom_x + zoom_y) * 72.0 / DPI / buffer
        imagebox = OffsetImage(cropped_sub_image, zoom=zoom)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (4, 1.2),
                xybox=(120., -80.),
                xycoords='data',
                pad=0.5,
                arrowprops=dict(
                    arrowstyle="->",
                    # connectionstyle="arc3,rad=0.3"
                    )
                )

        ax.add_artist(ab)
        annotation_bboxes.append(ab)



    fig.canvas.draw()
    for idx, proposed_configuration in enumerate(generate_proposed_tiles(TILE_N_X, TILE_N_Y, ax, annotation_bboxes)):
        apply_tile_configuration(proposed_configuration)

        plt.show()
        badness = measure_configuration_badness(main_line, proposed_configuration, fig.canvas.get_renderer())
        # TODO - function to check "badness" of the proposed arrangement (overlap, length of annotation, etc)
        fig.canvas.draw()

        # TEMP!
        

        fig_fn = rf"c:\temp\Prop-{idx}.png"
        plt.savefig(fig_fn, dpi=2*DPI, bbox_inches='tight',)
        print(f"{fig_fn} - {badness}")
        # plt.show()


if __name__ == "__main__":

    make_test_plot()


    for x in generate_proposed_tiles(3, 2, None, ["A", "B"]):
        print(x)