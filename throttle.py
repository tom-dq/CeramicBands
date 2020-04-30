

"""Throttling controls the elements which get updated at a given iteration, rather than just updating all of them"""
import enum

import typing

from common_types import ElemVectorDict, T_Elem_Axis


class StoppingCriterion(enum.Enum):
    element_count = enum.auto()
    new_prestrain_total = enum.auto()


class Shape(enum.Enum):
    step = enum.auto()    # Anything before cutoff is fully in, anything after makes not contribution
    linear = enum.auto()  # Taper from full contribution for the highest ranked element, linearly down to 0.0 for the last one to make it through cutoff.


class ElemStrainIncreaseData(typing.NamedTuple):
    elem_num: int
    axis: int
    proposed_prestrain_val: float
    old_prestrain_val: float


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
            increased_prestrains: typing.Dict[T_Elem_Axis, float],
            minor_prev_prestrain: ElemVectorDict,

    ) -> float:

        raise NotImplementedError


