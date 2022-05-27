"""Grab the same little snapshot from a series of images"""

import os
import pathlib

from PIL import Image

BASE_IMAGE_PATH = r"C:\Users\Tom Wilson\Documents\CeramicBandData\pics\4A"
BASE_PATH = pathlib.Path(BASE_IMAGE_PATH)
OUT_CROP_DIR = "cropped"

CROP_BOX = (100, 100, 200, 200)

def make_subfigs():
    existing_pngs = BASE_PATH.glob("*.png")

def crop_one_image(base_image: pathlib.Path, out_image: pathlib.Path):
    im = Image.open(base_image)

    im_cropped = im.crop()

if __name__ == "__main__":
    crop_one_image(BASE_PATH / "Case-0425.png", None)