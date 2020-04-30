

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


class ElemStrainIncreaseData(typing.NamedTuple):
    elem_num: int
    axis: int
    proposed_prestrain_val: float
    old_prestrain_val: float

    def proposed_abs_increase(self) -> float:
        increase = self.proposed_prestrain_val - self.old_prestrain_val
        return abs(increase)

    def ranking_with_priority(self, existing_prestrain_priority_factor: float) -> float:
        """Determines the priority of this strain increase for ranking purposes."""
        return self.proposed_abs_increase() + existing_prestrain_priority_factor * abs(self.old_prestrain_val)

    def throttle_contribution_full(
            self,
            init_data: InitialSetupModelData,
            stopping_criterion: StoppingCriterion
    ):
        if stopping_criterion == StoppingCriterion.new_prestrain_total:
            return self.proposed_abs_increase()

        elif stopping_criterion == StoppingCriterion.volume_ratio:
            return init_data.elem_volume[self.elem_num] / 2  # Divide by two since we may do X and Y prestrains

        else:
            raise ValueError(stopping_criterion)


class Throttler:
    stopping_criterion: StoppingCriterion = None
    shape: Shape = None
    cutoff_value: float = None

    def __init__(self, stopping_criterion: StoppingCriterion, shape: Shape, cutoff_value: float):

        if stopping_criterion == StoppingCriterion.new_prestrain_total and shape == Shape.linear:
            raise ValueError("Invalid combination - no unique solution of throttled values may exist.")

        self.stopping_criterion = stopping_criterion
        self.shape = shape
        self.cutoff_value = cutoff_value

    def throttle(
            self,
            init_data: InitialSetupModelData,
            increased_prestrains: typing.List[ElemStrainIncreaseData],
    ) -> typing.Iterable[ElemStrainIncreaseData]:

        if self.shape == Shape.step:
            accumulate
            takewhile

        # TODO - this is just the old behaviour

