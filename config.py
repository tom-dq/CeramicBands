import enum
import typing
import pathlib

import st7

class Environment(enum.Enum):
    uni_desktop = enum.auto()
    samsung_laptop = enum.auto()


class Config(typing.NamedTuple):
    fn_st7_base: pathlib.Path
    fn_working_image_base: pathlib.Path
    screenshot_res: st7.CanvasSize
    scratch_dir: pathlib.Path


def _get_env() -> Environment:
    return Environment.uni_desktop


def _get_config():
    this_env = _get_env()

    if this_env == Environment.uni_desktop:
        return Config(
            fn_st7_base=pathlib.Path(r"E:\Simulations\CeramicBands\v3\Test 9C-Contact-SD2.st7"),
            fn_working_image_base=pathlib.Path(r"E:\Simulations\CeramicBands\v4"),
            screenshot_res=st7.CanvasSize(1920, 1080),
            #screenshot_res=st7.CanvasSize(3840, 2160),
            scratch_dir=pathlib.Path(r"C:\Temp"),
        )

    elif this_env == Environment.samsung_laptop:
        pass



config = _get_config()
