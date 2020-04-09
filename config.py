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

    def summary_strings(self) -> typing.Iterable[str]:
        yield "Config:\n"
        for field_name, field_type in self._field_types.items():
            field_val = getattr(self, field_name)
            output_str = str(field_val)
            yield f"{field_name}\t{output_str}\n"

        yield "\n"


def _get_env() -> Environment:
    return Environment.uni_desktop


def _get_config():
    this_env = _get_env()

    if this_env == Environment.uni_desktop:
        return Config(
            fn_st7_base=pathlib.Path(r"E:\Simulations\CeramicBands\v5\Test 11-SD2.st7"),
            fn_working_image_base=pathlib.Path(r"E:\Simulations\CeramicBands\v5\pics"),
            screenshot_res=st7.CanvasSize(1920, 1080),
            #screenshot_res=st7.CanvasSize(3840, 2160),
            scratch_dir=pathlib.Path(r"C:\Temp"),
            solver=st7.SolverType.stQuasiStatic,
            qsa_steps_per_file=50,
            qsa_time_step_size=0.1,
        )

    elif this_env == Environment.samsung_laptop:
        pass



active_config = _get_config()
