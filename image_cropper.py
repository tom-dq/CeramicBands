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

def get_dilation_region(image: Image):

    crop_dims = _find_pink_pixels_crop_dims(image)
    image_cropped = image.crop(crop_dims)

    return image_cropped
    
    image_cropped.save(f"c:\\temp\\test_crop.png")

    # Find bounds of sub_image (no legend!)
    # Find min/max x and y of MAXED_OUT_PINK
    # Return that image
    ...


if __name__ == "__main__":
    image = Image.open(_test_image)
    get_dilation_region(image)

