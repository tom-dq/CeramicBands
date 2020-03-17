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
    solver: st7.SolverType
    qsa_steps_per_file: int
    qsa_time_step_size: float


def _get_env() -> Environment:
    return Environment.uni_desktop


def _get_config():
    this_env = _get_env()

    if this_env == Environment.uni_desktop:
        return Config(
            fn_st7_base=pathlib.Path(r"E:\Simulations\CeramicBands\v5\Test 10-Contact.st7"),
            fn_working_image_base=pathlib.Path(r"E:\Simulations\CeramicBands\v5\pics"),
            screenshot_res=st7.CanvasSize(1920, 1080),
            #screenshot_res=st7.CanvasSize(3840, 2160),
            scratch_dir=pathlib.Path(r"C:\Temp"),
            solver=st7.SolverType.stQuasiStaticSolver,
            qsa_steps_per_file=20,
            qsa_time_step_size=0.1,
        )

    elif this_env == Environment.samsung_laptop:
        pass



active_config = _get_config()
