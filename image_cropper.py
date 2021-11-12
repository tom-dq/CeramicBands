from PIL import Image, ImageDraw, ImageFont
import numpy

MAXED_OUT_PINK = (217, 26, 217)


def find_dilation_region(image: Image):

    ar = numpy.array(image)

    # Find bounds of sub_image (no legend!)
    # Find min/max x and y of MAXED_OUT_PINK
    # Return that image
    ...

