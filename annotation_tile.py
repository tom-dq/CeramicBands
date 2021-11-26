"""Utility for tiling subfigures in a matplotlib figure in the "least bad" way."""

import itertools
import math
import typing
import pathlib
import time


import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.markers
import matplotlib.lines

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.transforms import Bbox

from PIL import Image

from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox

import image_cropper

DPI = 150


class ProposedTile(typing.NamedTuple):
    i_x: int
    i_y: int
    annotation_bbox: AnnotationBbox


class ProposedConfiguration(typing.NamedTuple):
    n_x: int
    n_y: int
    ax: Axes
    tiles: typing.List[ProposedTile]

    def get_bbox_axis_fraction_limits(self, proposed_tile: ProposedTile) -> Bbox:

        xmin, xmax = [(proposed_tile.i_x + ii) / self.n_x for ii in (0, 1)]
        ymin, ymax = [(proposed_tile.i_y + ii) / self.n_y for ii in (0, 1)]
        
        data = [[xmin, ymin], [xmax, ymax]]
        return Bbox(data)

    def get_bbox_axis_data_limits(self, ax: Axes, proposed_tile: ProposedTile) -> Bbox:

        bbox_axes_fraction = self.get_bbox_axis_fraction_limits(proposed_tile)

        # There must be a better way to do this!
        xmin = ax.dataLim.x0 + ax.dataLim.width*bbox_axes_fraction.x0
        xmax = ax.dataLim.x0 + ax.dataLim.width*bbox_axes_fraction.x1

        ymin = ax.dataLim.y0 + ax.dataLim.height*bbox_axes_fraction.y0
        ymax = ax.dataLim.y0 + ax.dataLim.height*bbox_axes_fraction.y1

        bbox_data = Bbox([[xmin, ymin], [xmax, ymax]])

        return bbox_data


class AssessedConfiguration(typing.NamedTuple):
    intersections: int
    total_arrow_distance: float
    proposed_configuration: ProposedConfiguration


def generate_proposed_tiles(n_x: int, n_y: int, only_edges: bool, ax: Axes, main_lines, annotation_bboxes: typing.List[AnnotationBbox], ) -> typing.Iterable[ProposedConfiguration]:
    

    all_tiles = list(itertools.product(range(n_x), range(n_y)))

    # If needed, only consider the tiles around the perimeter
    if only_edges:
        edge_filtered_tiles = [(i_x, i_y) for (i_x, i_y) in all_tiles if i_x in (0, n_x-1) or i_y in (0, n_y-1)]

    else:
        edge_filtered_tiles = all_tiles

    # Remove any tiles which would intersect
    non_intersecting_tiles = []
    for i_x, i_y in edge_filtered_tiles:
        proposed_tile = ProposedTile(i_x=i_x, i_y=i_y, annotation_bbox=...)
        dummy_proposed_config = ProposedConfiguration(n_x, n_y, ax, [])
        bbox_data = dummy_proposed_config.get_bbox_axis_data_limits(ax, proposed_tile)
        intersect_free = True
        for main_line in main_lines:
            line_path = main_line.get_path()
            if line_path.intersects_bbox(bbox_data):
                intersect_free = False

        if intersect_free:
            non_intersecting_tiles.append((i_x, i_y))

    # Preflight check - how many options are we dealing with?
    n_perms = math.perm(len(all_tiles), len(annotation_bboxes))
    n_perms_non_intersecting = math.perm(len(non_intersecting_tiles), len(annotation_bboxes))
    print(f"Total options on the table: {n_perms}")
    print(f"After filtering intersections: {n_perms_non_intersecting}")

    for tile_proposal in itertools.permutations(non_intersecting_tiles, len(annotation_bboxes)):
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
        bbox_axes_fraction = proposed_configuration.get_bbox_axis_fraction_limits(proposed_tile)

        new_xybox = (
            0.5 * (bbox_axes_fraction.xmin + bbox_axes_fraction.xmax),
            0.5 * (bbox_axes_fraction.ymin + bbox_axes_fraction.ymax),
        )

        proposed_tile.annotation_bbox.xybox = new_xybox
        proposed_tile.annotation_bbox.boxcoords = 'axes fraction'


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


def assess_configuration_badness(
        ax: Axes,
        main_lines: typing.List[matplotlib.lines.Line2D],
        proposed_configuration: ProposedConfiguration,
    ) -> AssessedConfiguration:

    line_paths = [main_line.get_path() for main_line in main_lines]

    intersects = 0
    total_distance = 0.0
    for proposed_tile in proposed_configuration.tiles:

        bbox_data = proposed_configuration.get_bbox_axis_data_limits(ax, proposed_tile)

        # Distance between figure and annotation point...
        fig_x = 0.5 * (bbox_data.xmin + bbox_data.xmax)
        fig_y = 0.5 * (bbox_data.ymin + bbox_data.ymax)

        d_x = (fig_x - proposed_tile.annotation_bbox.xy[0])
        d_y = (fig_y - proposed_tile.annotation_bbox.xy[1])
        dist = math.sqrt(d_x**2 + d_y**2)
        total_distance += dist

        for line_path in line_paths:
            line_intersects = line_path.intersects_bbox(bbox_data)
            if line_intersects:
                intersects += 1

    return AssessedConfiguration(intersects, total_distance, proposed_configuration)
    

def save_best_configuration_to(ax, main_lines, proposed_configurations: typing.Iterable[ProposedConfiguration], fn_out: str):

    assessed_configurations = [assess_configuration_badness(ax, main_lines, proposed_configuration) for proposed_configuration in proposed_configurations]

    assessed_configurations.sort()

    best_configuration = assessed_configurations[0].proposed_configuration

    apply_tile_configuration(best_configuration)
    plt.savefig(fn_out, dpi=2*DPI, bbox_inches='tight',)



def make_test_plot():
    screen_h = 2160

    TILE_N_X, TILE_N_Y = 3, 4

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

    main_lines = [main_line]

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

    proposed_configurations = generate_proposed_tiles(TILE_N_X, TILE_N_Y, ax, main_lines, annotation_bboxes)
    save_best_configuration_to(
        ax,
        main_lines,
        proposed_configurations,
        rf"c:\temp\bad_checks\Best.png",
    )

if __name__ == "__main__":

    make_test_plot()

