import itertools
import pathlib

from PIL import Image


MAXED_OUT_PINK = (217, 26, 217)

_test_image = r"C:\Users\Tom Wilson\Documents\CeramicBandData\outputs\192.168.1.109+8080\v7\pics\DR\Case-1257.png"

def _find_pink_pixels_crop_dims(image: Image):
    width, height = image.size

    # Don't use the first 10 percent so the legend is excluded
    width_min = int(0.1*width)

    def get_all_pink_pixels():
        pixels = image.load()
        for x, y in itertools.product(range(width_min, width), range(height)):
            if pixels[x, y] == MAXED_OUT_PINK:
                yield x, y

    x_pixels = set()
    y_pixels = set()
    for x, y in get_all_pink_pixels():
        x_pixels.add(x)
        y_pixels.add(y)

    buffer = int(width / 100)

    return (
        min(x_pixels) - buffer,
        min(y_pixels) - buffer,
        max(x_pixels) + buffer,
        max(y_pixels) + buffer,
    )

def get_dilation_region(target_aspect_ratio: float, image: Image):

    # Get the full image with "pink dilation"
    crop_dims = _find_pink_pixels_crop_dims(image)

    # Crop that to the right aspect ratio
    x_range = crop_dims[2] - crop_dims[0]
    y_range = crop_dims[3] - crop_dims[1]
    
    full_region_aspect = x_range / y_range
    if target_aspect_ratio < full_region_aspect:
        # Need to remove the ends in X
        space_for_x = target_aspect_ratio * y_range
        removed_x = int((x_range - space_for_x) / 2)
        cropped_aspect_dims = (
            crop_dims[0] + removed_x,
            crop_dims[1],
            crop_dims[2] - removed_x,
            crop_dims[3],
        )

    else:
        # Need to remove the ends in Y
        space_for_y = x_range / target_aspect_ratio
        removed_y = int((y_range - space_for_y) / 2)

        cropped_aspect_dims = (
            crop_dims[0],
            crop_dims[1] + removed_y,
            crop_dims[2],
            crop_dims[3] - removed_y,
        )

    image_cropped = image.crop(cropped_aspect_dims)

    return image_cropped
    
    image_cropped.save(f"c:\\temp\\test_crop.png")

    # Find bounds of sub_image (no legend!)
    # Find min/max x and y of MAXED_OUT_PINK
    # Return that image
    ...


if __name__ == "__main__":
    image = Image.open(_test_image)
    get_dilation_region(1.2, image)

