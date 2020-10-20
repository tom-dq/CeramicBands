

"""Throttling controls the elements which get updated at a given iteration, rather than just updating all of them"""
import abc
import collections
import enum

import typing

import numpy
from numpy.linalg.linalg import eig

from common_types import ElemVectorDict, SingleValue, T_Elem_Axis, InitialSetupModelData, func_repr, Actuator
from tables import Table

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


class ElemPreStrainChangeData(typing.NamedTuple):
    elem_num: int
    axis: int
    proposed_prestrain_val: float
    old_prestrain_val: float
    result_strain_val: float
    eigen_vector_old: typing.Optional[numpy.array]
    eigen_vector_proposed: typing.Optional[numpy.array]
    elem_volume_ratio: float

    def scaled_down(self, factor: float) -> "ElemPreStrainChangeData":
        """Taper off the full value."""

        if factor > 1 or factor < 0:
            raise ValueError(factor)

        full_delta = self.proposed_prestrain_val - self.old_prestrain_val
        scaled_delta = factor * full_delta

        new_proposed_prestrain = self.old_prestrain_val + scaled_delta

        # Scale down the eigenvector, if needs be
        if self.eigen_vector_old is None:
            new_eigen_proposed = self.eigen_vector_proposed

        else:
            # Have to actually scale the eigenvector. Use the angle proportional to the value scaling delta

            eigen_vector_delta = self.eigen_vector_proposed - self.eigen_vector_old
            eigen_vector_new = self.eigen_vector_old + (factor*eigen_vector_delta)

            norm = numpy.linalg.norm(eigen_vector_new)
            new_eigen_proposed = eigen_vector_new / norm

        return self._replace(proposed_prestrain_val=new_proposed_prestrain, eigen_vector_proposed=new_eigen_proposed)

    def proposed_change(self) -> float:
        return self.proposed_prestrain_val - self.old_prestrain_val

    def proposed_abs_increase(self) -> float:
        """If the prestrain went further away from zero (on the same side), how much was that?"""

        # Sign change -> no abs increase
        if (self.old_prestrain_val * self.proposed_prestrain_val) < 0:
            return 0.0

        maybe_abs_increase = abs(self.proposed_prestrain_val) - abs(self.old_prestrain_val)
        if maybe_abs_increase > 0:
            return maybe_abs_increase

        return 0.0


    def ranking_with_priority(self, existing_prestrain_priority_factor: float) -> float:
        """Determines the priority of this strain increase for ranking purposes."""

        return (
                existing_prestrain_priority_factor * abs(self.old_prestrain_val) + # Optionally boost elements which were already prestrained.
                abs(self.proposed_prestrain_val) +
                self.result_strain_val
        )

        return (
            existing_prestrain_priority_factor * abs(self.old_prestrain_val) +  # Optionally boost elements which were already prestrained.
            self.result_strain_val + abs(self.old_prestrain_val) +  # This is the old "total" strain.
            self.proposed_abs_increase()
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

    @staticmethod
    def princ_transform_scale(actuator: Actuator, factor: float, all_epscd: typing.Iterable["ElemPreStrainChangeData"]):
        """Does the scale-down on the principal axis"""

        # Scales down the magnitusde and the angle...

        OLD, NEW = 5, 6
        
        elem_to_data = collections.defaultdict(dict)
        for epscd in all_epscd:
            for key, val in ( (OLD, epscd.old_prestrain_val), (NEW, epscd.new_proposed_prestrain)):
                pass

        raise ValueError("Up to here....")

    def to_single_value(self) -> SingleValue:
        return SingleValue(
            elem=self.elem_num,
            axis=self.axis,
            value=self.proposed_prestrain_val,
            eigen_vector=self.eigen_vector_proposed,
        )

    def vol_scaled_prestrain_contrib(self) -> float:
        return self.elem_volume_ratio * self.proposed_prestrain_val

class BaseThrottler:
    @abc.abstractmethod
    def throttle(
            self,
            init_data: InitialSetupModelData,
            run_params,  # This is main.RunParams
            proposed_prestrains: typing.List[ElemPreStrainChangeData],
    ) -> typing.List[ElemPreStrainChangeData]:
        raise NotImplementedError()


class Throttler(BaseThrottler):
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
            increased_prestrains: typing.List[ElemPreStrainChangeData],
    ) -> typing.Tuple[float, typing.List[ElemPreStrainChangeData]]:
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
            increased_prestrains: typing.List[ElemPreStrainChangeData],
    ) -> typing.List[ElemPreStrainChangeData]:

        running_total, sub_list = self._running_total_and_list(True, init_data, increased_prestrains)
        return sub_list

    def throttle(
            self,
            init_data: InitialSetupModelData,
            run_params,  # This is main.RunParams
            proposed_prestrains: typing.List[ElemPreStrainChangeData],
    ) -> typing.List[ElemPreStrainChangeData]:

        proposed_prestrains.sort(
            reverse=True,
            key=lambda epscd: epscd.ranking_with_priority(run_params.existing_prestrain_priority_factor)
        )

        # Limit to cases where the pre-strain is increasing
        increasing_prestrains = [epscd for epscd in proposed_prestrains if epscd.proposed_abs_increase() > 0]

        if self.shape == Shape.step:
            return self._go_to_cutoff_step(init_data, increasing_prestrains)

        elif self.shape == Shape.linear:
            return self._find_tapered_scale_end(init_data, increasing_prestrains)

        else:
            raise ValueError(self.shape)

    def _find_tapered_scale_end(
            self,
            init_data: InitialSetupModelData,
            increased_prestrains: typing.List[ElemPreStrainChangeData],
    ) -> typing.List[ElemPreStrainChangeData]:
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
            increased_prestrains: typing.List[ElemPreStrainChangeData],
            idx_end: int
    ) -> typing.Iterable[ElemPreStrainChangeData]:
        """Linearly scaled down the pre-strain increases."""

        for idx, one in enumerate(increased_prestrains[:idx_end]):
            scale_down_factor = (len(increased_prestrains) - idx) / len(increased_prestrains)
            yield one.scaled_down(scale_down_factor)


class RelaxedIncreaseDecrease(BaseThrottler):
    """Let all the elements increase or decrease as the iteration progresses, but don't always go the full
        proposed amount."""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    def throttle(
            self,
            init_data: InitialSetupModelData,
            previous_prestrain_update,
            run_params,  # This is main.RunParams
            proposed_prestrains: typing.List[ElemPreStrainChangeData],
    ) -> typing.List[ElemPreStrainChangeData]:

        # First scaling - element by element based.
        ratio = run_params.parameter_trend.throttler_relaxation(run_params.parameter_trend.current_inc)

        scaled_down_proposed_strains = [epscd.scaled_down(ratio) for epscd in proposed_prestrains]

        # Second scaling - limit the "overall" prestrain scaling to some limit.
        total_this_iter = sum(epscd.vol_scaled_prestrain_contrib() for epscd in proposed_prestrains)
        overall_delta = total_this_iter - previous_prestrain_update.overall_dilation_ratio_working_set
        limit_delta = run_params.parameter_trend.overall_iterative_prestrain_delta_limit(run_params.parameter_trend.current_inc)
        if abs(overall_delta) > limit_delta:
            print("Scaling down!")

            scaled_total_delta = abs(limit_delta / overall_delta)

            final_scale = [epscd.scaled_down(scaled_total_delta) for epscd in proposed_prestrains]

        else:
            final_scale = proposed_prestrains

        return final_scale
