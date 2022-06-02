"""Grab the same little snapshot from a series of images"""

import os
import pathlib

from PIL import Image

BASE_IMAGE_PATH = r"C:\Users\Tom Wilson\Documents\CeramicBandData\pics\4A"
BASE_PATH = pathlib.Path(BASE_IMAGE_PATH)
OUT_CROP_DIR = "cropped"

L, W = 1208, 141
CROP_BOX = (L, 66, 2560-(0), 1440-66-322)

def make_subfigs():
    existing_pngs = BASE_PATH.glob("*.png")

    out_path_base = BASE_PATH / OUT_CROP_DIR
    os.makedirs(out_path_base, exist_ok=True)


def crop_one_image(base_image: pathlib.Path, out_image: pathlib.Path):
    im = Image.open(base_image)

    im_cropped = im.crop(CROP_BOX)

    im_cropped.save(out_image)


if __name__ == "__main__":
    make_subfigs()
    out_path = BASE_PATH / OUT_CROP_DIR / "AA-Case-0425.png"
    crop_one_image(BASE_PATH / "Case-0425.png", out_path)