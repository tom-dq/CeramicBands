import enum
import typing
import pathlib
import platform

from st7_wrap import st7
from st7_wrap import const

class Environment(enum.Enum):
    uni_desktop = enum.auto()
    samsung_laptop = enum.auto()
    macbook_bootcamp = enum.auto()


class MaxIters(typing.NamedTuple):
    """Either both of these are set, or neither."""
    major_step: int
    minor_step: int


class HistoryToRecord(enum.Enum):
    none = enum.auto()
    limited = enum.auto()
    full = enum.auto()

    @property
    def only_deformed_node_pos(self) -> bool:
        if self == HistoryToRecord.limited:
            return True

        elif self == HistoryToRecord.full:
            return False

        else:
            raise ValueError(self)


class Config(typing.NamedTuple):
    fn_st7_base: pathlib.Path
    fn_working_image_base: pathlib.Path
    screenshot_res: st7.CanvasSize
    scratch_dir: pathlib.Path
    solver: const.SolverType
    qsa_steps_per_file: int
    qsa_time_step_size: float
    max_iters: typing.Optional[MaxIters]
    delete_old_result_files: bool
    ratched_prestrains_during_iterations: bool
    case_for_every_increment: bool
    record_result_history_in_db: HistoryToRecord
    converged_delta_prestrain: float


    def summary_strings(self) -> typing.Iterable[str]:
        yield "Config:\n"
        for field_name, field_type in self.__annotations__.items():
            field_val = getattr(self, field_name)
            output_str = str(field_val)
            yield f"{field_name}\t{output_str}\n"

        yield "\n"


def _get_env() -> Environment:
    if platform.node() == "DESKTOP-EFCE6TR":
        return Environment.macbook_bootcamp

    elif platform.node() == "DESKTOP-5D1C7NO":
        return Environment.uni_desktop

    else:
        raise ValueError(f"Unknown host: {platform.node()}")


def _get_config():
    this_env = _get_env()

    if this_env == Environment.uni_desktop:
        return Config(
            # fn_st7_base=pathlib.Path(r"E:\Simulations\CeramicBands\v5\Test 12-SingleGrade.st7"),
            # fn_st7_base=pathlib.Path(r"E:\Simulations\CeramicBands\v5\Test 11.st7"),
            fn_st7_base=pathlib.Path(r"D:\Tom Wilson\Dropbox\Dropbox\PhD\Ceramic Bands Source Models\v7-Wedge"),
            fn_working_image_base=pathlib.Path(r"E:\Simulations\CeramicBands\v7\pics"),
            # screenshot_res=st7.CanvasSize(1920, 1080),  # FHD
            # screenshot_res=st7.CanvasSize(2560, 1440),  # QHD
            screenshot_res=st7.CanvasSize(3840, 2160), # 4K
            scratch_dir=pathlib.Path(r"C:\Temp"),
            solver=const.SolverType.stQuasiStatic,
            qsa_steps_per_file=10,
            qsa_time_step_size=0.1,
            max_iters=MaxIters(major_step=5, minor_step=1),
            delete_old_result_files=True,
            ratched_prestrains_during_iterations=False,
            case_for_every_increment=False,
            record_result_history_in_db=HistoryToRecord.limited,
            converged_delta_prestrain=1e-6,
        )

    elif this_env == Environment.macbook_bootcamp:
        return Config(
            fn_st7_base=pathlib.Path(r"C:\Users\Tom Wilson\Dropbox\PhD\Ceramic Bands Source Models\v7-Wedge"),
            fn_working_image_base=pathlib.Path(r"C:\Users\Tom Wilson\Documents\CeramicBandData\pics"),
            screenshot_res=st7.CanvasSize(2560, 1440),  # QHD
            scratch_dir=pathlib.Path(r"C:\Temp"),
            solver=const.SolverType.stQuasiStatic,
            qsa_steps_per_file=10,
            qsa_time_step_size=0.1,
            max_iters=MaxIters(major_step=5, minor_step=1),
            delete_old_result_files=True,
            ratched_prestrains_during_iterations=False,
            case_for_every_increment=False,
            record_result_history_in_db=HistoryToRecord.limited,
            converged_delta_prestrain=1e-6,
        )

    elif this_env == Environment.samsung_laptop:
        pass



active_config = _get_config()
