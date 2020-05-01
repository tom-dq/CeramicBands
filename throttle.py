

"""Throttling controls the elements which get updated at a given iteration, rather than just updating all of them"""
import enum

import typing

from common_types import ElemVectorDict, T_Elem_Axis, InitialSetupModelData


class StoppingCriterion(enum.Enum):
    volume_ratio = enum.auto()
    new_prestrain_total = enum.auto()


class Shape(enum.Enum):
    step = enum.auto()    # Anything before cutoff is fully in, anything after makes not contribution
    linear = enum.auto()  # Taper from full contribution for the highest ranked element, linearly down to 0.0 for the last one to make it through cutoff.


class ThrottleMethod(enum.Enum):
    volume_step = enum.auto()
    volume_linear = enum.auto()
    prestrain_step = enum.auto()

    @property
    def shape(self) -> Shape:
        if self in (ThrottleMethod.volume_step, ThrottleMethod.prestrain_step):
            return Shape.step

        elif self == ThrottleMethod.volume_linear:
            return Shape.linear

        else:
            raise ValueError(self)


    @property
    def stopping_criterion(self) -> StoppingCriterion:
        if self in (ThrottleMethod.volume_step, ThrottleMethod.volume_linear):
            return StoppingCriterion.volume_ratio

        elif self == ThrottleMethod.prestrain_step:
            return StoppingCriterion.new_prestrain_total

        else:
            raise ValueError(self)


class ElemStrainIncreaseData(typing.NamedTuple):
    elem_num: int
    axis: int
    proposed_prestrain_val: float
    old_prestrain_val: float
    result_strain_val: float

    def scaled_down(self, factor: float) -> "ElemStrainIncreaseData":
        """Taper off the full value."""

        if factor > 1 or factor < 0:
            raise ValueError(factor)

        full_delta = self.proposed_prestrain_val - self.old_prestrain_val
        scaled_delta = factor * full_delta

        new_proposed_prestrain = self.old_prestrain_val + scaled_delta

        return self._replace(proposed_prestrain_val=new_proposed_prestrain)

    def proposed_abs_increase(self) -> float:
        increase = self.proposed_prestrain_val - self.old_prestrain_val
        return abs(increase)

    def ranking_with_priority(self, existing_prestrain_priority_factor: float) -> float:
        """Determines the priority of this strain increase for ranking purposes."""

        return (
            existing_prestrain_priority_factor * abs(self.old_prestrain_val) +  # Optionally boost elements which were already prestrained.
            self.result_strain_val + abs(self.old_prestrain_val)  # This is the old "total" strain.
        )

    def throttle_contribution_full(
            self,
            init_data: InitialSetupModelData,
            stopping_criterion: StoppingCriterion
    ):
        if stopping_criterion == StoppingCriterion.new_prestrain_total:
            return self.proposed_abs_increase() * init_data.elem_volume_ratio[self.elem_num] / 2

        elif stopping_criterion == StoppingCriterion.volume_ratio:
            return init_data.elem_volume_ratio[self.elem_num] / 2  # Divide by two since we may do X and Y prestrains

        else:
            raise ValueError(stopping_criterion)


class Throttler:
    stopping_criterion: StoppingCriterion = None
    shape: Shape = None
    cutoff_value: float = None

    def __init__(self, stopping_criterion: StoppingCriterion, shape: Shape, cutoff_value: float):
        self.stopping_criterion = stopping_criterion
        self.shape = shape
        self.cutoff_value = cutoff_value

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.stopping_criterion}, {self.shape}, {self.cutoff_value})"

    def _running_total_and_list(
            self,
            stop_at_cutoff: bool,
            init_data: InitialSetupModelData,
            increased_prestrains: typing.List[ElemStrainIncreaseData],
    ) -> typing.Tuple[float, typing.List[ElemStrainIncreaseData]]:
        working_list = []

        running_total = 0.0

        for one in increased_prestrains:
            working_list.append(one)
            running_total += one.throttle_contribution_full(init_data, self.stopping_criterion)

            if (running_total > self.cutoff_value) and stop_at_cutoff:
                break

        return running_total, working_list

    def _go_to_cutoff_step(
            self,
            init_data: InitialSetupModelData,
            increased_prestrains: typing.List[ElemStrainIncreaseData],
    ) -> typing.List[ElemStrainIncreaseData]:

        running_total, sub_list = self._running_total_and_list(True, init_data, increased_prestrains)
        return sub_list

    def throttle(
            self,
            init_data: InitialSetupModelData,
            increased_prestrains: typing.List[ElemStrainIncreaseData],
    ) -> typing.List[ElemStrainIncreaseData]:

        running_total = 0.0

        if self.shape == Shape.step:
            return self._go_to_cutoff_step(init_data, increased_prestrains)

        elif self.shape == Shape.linear:
            return self._find_tapered_scale_end(init_data, increased_prestrains)

        else:
            raise ValueError(self.shape)

    def _find_tapered_scale_end(
            self,
            init_data: InitialSetupModelData,
            increased_prestrains: typing.List[ElemStrainIncreaseData],
    ) -> typing.List[ElemStrainIncreaseData]:
        lower_bound = 1
        upper_bound = len(increased_prestrains)

        def have_found() -> bool:
            if abs(lower_bound - upper_bound) < 2:
                return True

            return False

        increased_prestrains_head = None

        while not have_found():
            mid_guess = (lower_bound + upper_bound) // 2
            increased_prestrains_guess = increased_prestrains[0:mid_guess]
            guess_total, increased_prestrains_head = self._running_total_and_list(False, init_data, increased_prestrains_guess)

            print(f"lower={lower_bound}\tupper={upper_bound}\tat [{mid_guess}], total={guess_total}\ttarget={self.cutoff_value}")

            if guess_total > self.cutoff_value:
                # Have too much - need a shorter list.
                upper_bound = mid_guess

            else:
                # Don't have enough - need to go up
                lower_bound = mid_guess

        # In case nothing ever ran.
        if not increased_prestrains_head:
            mid_guess = (lower_bound + upper_bound) // 2
            increased_prestrains_guess = increased_prestrains[0:mid_guess]
            _, increased_prestrains_head = self._running_total_and_list(False, init_data, increased_prestrains_guess)

        return increased_prestrains_head

    def _tapered_scale(
            self,
            increased_prestrains: typing.List[ElemStrainIncreaseData],
            idx_end: int
    ) -> typing.Iterable[ElemStrainIncreaseData]:
        """Linearly scaled down the pre-strain increases."""

        for idx, one in enumerate(increased_prestrains[:idx_end]):
            scale_down_factor = (len(increased_prestrains) - idx) / len(increased_prestrains)
            yield one.scaled_down(scale_down_factor)



