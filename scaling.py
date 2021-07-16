import abc
from ast import Num
import collections
import math
import random
import typing
import bisect

import parameter_trend

from st7_wrap import st7

from common_types import SingleValue, T_ScaleKey, InitialSetupModelData, ElemVectorDict


class Scaling:

    _adjacent_elements: typing.Dict[int, typing.Dict[int, float]]  # Elem -> Adj_Elem -> Weight
    _working_prestrain_vals: typing.Dict[T_ScaleKey, float]
    _parameter_trend: parameter_trend.ParameterTrend

    """Scales the "Yield Stress" according to some criterion."""
    @abc.abstractmethod
    def get_x_scale_factor(self, scale_key: T_ScaleKey) -> float:
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__

    def assign_centroids(self, init_data: InitialSetupModelData):
        """This is not used by all the methods."""
        pass

    def assign_working_results(self, previous_iteration_results: typing.List[SingleValue]):
        self._working_prestrain_vals = { sv.id_key: sv.value for sv in previous_iteration_results }

    def determine_adjacency(self, init_data: InitialSetupModelData):
        """Weighted element-to-neighbour. One shared node -> 1/16 contrib. Two shared nodes -> 1/8 contrib.
        Include the element itself."""

        # TODO - add "phantom elements" which are reflected over free boundaries.
        node_to_elems = collections.defaultdict(set)
        for elem, conn in init_data.elem_conns.items():
            for node in conn:
                node_to_elems[node].add(elem)

        adj_elems = {}
        for elem, conn in init_data.elem_conns.items():
            this_elem_weighs = collections.Counter()
            for node in conn:
                adj_elements = node_to_elems[node].copy()

                for adj_element in adj_elements:
                    this_elem_weighs[adj_element] += 1

            weighted_contribs = {other_elem: count / 16 for other_elem, count in this_elem_weighs.items()}
            adj_elems[elem] = weighted_contribs

        self._adjacent_elements = adj_elems

    def _get_adj_elem_scale_factor(self, scale_key: T_ScaleKey) -> float:
        """Optionally give the input strains a boost if the neighbors have yielded.
            If all adjacent elements are fully dialated but this element is not dialated, and adj_strain_ratio = 1.0,
            has a "boost" factor of 0.75.


            adj_strain_ratio    boost_factor    return
            0                   0.75            1
            0.2                 0.75            0.789473684
            1                   0.75            0.428571429
            5                   0.75            0.130434783
            0                   0.2             1
            0.2                 0.2             0.5
            1                   0.2             0.166666667
            5                   0.2             0.038461538
            0                   0               1
            0.2                 0               1
            1                   0               1
            5                   0               1
            """

        adj_strain_ratio_true = self._parameter_trend.adj_strain_ratio_true(self._parameter_trend.current_inc)

        if not adj_strain_ratio_true:
            # Do not scale
            return 1.0

        full_dilation_ratio = self._parameter_trend.dilation_ratio(self._parameter_trend.current_inc)

        adj_elem_contribs = 0.0
        elem, direction = scale_key
        for adj_elem, neighbor_factor in self._adjacent_elements[elem].items():
            adj_scale_key = (adj_elem, direction)
            adj_elem_contribs += neighbor_factor * self._working_prestrain_vals.get(adj_scale_key, 0.0)

        boost_factor = adj_strain_ratio_true * abs(adj_elem_contribs) / full_dilation_ratio

        return 1.0 + boost_factor


class NoScaling(Scaling):
    def __init__(self):
        pass

    def get_x_scale_factor(self, scale_key: T_ScaleKey) -> float:
        return 1.0


class RandomScaling(Scaling):
    """Scales each element+direction randomly, making up the numbers as the elements turn up."""
    _x_scaler: dict
    _random_spread: float

    def __init__(self, random_spread: float):
        # We can scale the x requirements by a small amount to distribute the starting point.
        self._x_scaler = dict()
        self._random_spread = random_spread


    def get_x_scale_factor(self, scale_key: T_ScaleKey) -> float:
        parameter_trend_ratio = self._parameter_trend.scaling_ratio(self._parameter_trend.current_inc)

        if scale_key not in self._x_scaler:
            self._x_scaler[scale_key] = 1.0 + random.uniform(-1 * self._random_spread, self._random_spread)

        return parameter_trend_ratio * self._x_scaler[scale_key]

    def __str__(self):
        return f"{self.__class__.__name__}(random_spread={self._random_spread})"


class CentroidAwareScaling(Scaling):
    """Keeps track of where the elements are."""

    _elem_scale_fact: dict
    _spacing: float
    _amplitude: float
    _y_min: float
    _y_max: float
    _y_bottom: float
    _y_depth: float
    _y_no_yielding_below: float
    _x_cent: float

    # Put a thumb on the the scales and make it harder for the bigger elements (not the region of interest) to yield.
    INHIBIT_LARGE_ELEMENTS_YIELDING = True

    @abc.abstractmethod
    def _scale_factor_one_elem(self, cent: st7.Vector3):
        raise NotImplementedError()

    def _get_element_size_factors(self, init_data: InitialSetupModelData) -> typing.Dict[int, float]:
        min_elem_size = min(vol for vol in init_data.elem_volume.values())

        def scale_fact(vol):
            if self.INHIBIT_LARGE_ELEMENTS_YIELDING:
                cutoff_vol = 1.5 * min_elem_size
                clamped_vol = max(vol, cutoff_vol)

                return cutoff_vol / clamped_vol

            else:
                return 1.0

        return {elem_num: scale_fact(vol) for elem_num, vol in init_data.elem_volume.items()}

    def assign_centroids(self, init_data: InitialSetupModelData):
        elem_centroid = init_data.elem_centroid

        # Find out how deep the elements go.
        self._y_max = max(xyz.y for xyz in elem_centroid.values())
        self._y_bottom = min(xyz.y for xyz in elem_centroid.values())

        self._y_min = self._y_max - self._y_depth
        self._x_cent = 0.5 * (min(xyz.x for xyz in elem_centroid.values()) + max(xyz.x for xyz in elem_centroid.values()))

        hard_for_big_elements_to_yield = self._get_element_size_factors(init_data)

        for elem_num, elem_cent in elem_centroid.items():
            blank_out_bottom_elems = self._no_base_yield_modifier(elem_cent)
            self._elem_scale_fact[elem_num] = self._scale_factor_one_elem(elem_cent) * blank_out_bottom_elems * hard_for_big_elements_to_yield[elem_num]

        self.determine_adjacency(init_data)

    def get_element_surface_depth_ratio(self, cent: st7.Vector3):
        """1.0 right on the top or bottom surface, scaled down to 0.0 at _y_depth"""
        surface_depth = min(
            self._y_max - cent.y,
            cent.y - self._y_bottom,
        )

        if surface_depth > self._y_depth:
            return 0.0

        return (self._y_depth - surface_depth) / self._y_depth


    def get_x_scale_factor(self, scale_key: T_ScaleKey) -> float:
        """Higher number means EASIER for element to yield. For example, a return value of 2.0 means it takes half the load for this element to yield."""
        elem_num, _ = scale_key[:]
        dilated_neighbor_scale = self._get_adj_elem_scale_factor(scale_key)

        parameter_trend_ratio = self._parameter_trend.scaling_ratio(self._parameter_trend.current_inc)

        return 0.5 * (parameter_trend_ratio * self._elem_scale_fact[elem_num] + dilated_neighbor_scale)

    def _no_base_yield_modifier(self, elem_cent) -> float:
        # Deactive.
        return 1.0

        if elem_cent.y < self._y_no_yielding_below:
            return 0.0

        else:
            return 1.0


class CosineScaling(CentroidAwareScaling):
    """Does a cosine based on the element position."""

    def __init__(self, pt: parameter_trend.ParameterTrend, y_depth: float, spacing: float, amplitude: float):
        self._parameter_trend = pt
        self._elem_scale_fact = {}
        self._y_depth = y_depth
        self._spacing = spacing
        self._amplitude = amplitude

    def _scale_factor_one_elem(self, cent: st7.Vector3):
        x, _, _ = cent[:]

        real_amplitude = self.get_element_surface_depth_ratio(cent) * self._amplitude
        a = self._spacing / (math.pi * 2)
        raw_cosine = math.cos(a * x)

        return 1.0 + real_amplitude * raw_cosine

    def __str__(self):
        return f"{self.__class__.__name__}(spacing={self._spacing}, amplitude={self._amplitude}, y_depth={self._y_depth})"


class SpacedStepScaling(CentroidAwareScaling):
    """Mimic the regular spacing of the holes drilled in the surface."""

    _hole_width: float

    def __init__(self, pt: parameter_trend.ParameterTrend, y_depth: float, spacing: float, amplitude: float, hole_width: float, max_variation: float):
        self._elem_scale_fact = {}
        self._y_depth = y_depth
        self._spacing = spacing
        self._amplitude = amplitude
        self._hole_width = hole_width
        self._parameter_trend = pt
        self._max_variation = max_variation

        self._hole_cent_to_amp = self._get_hole_x_cent_and_amplitude()
        self._hole_cent_sorted = sorted(self._hole_cent_to_amp.keys())

    def _get_hole_and_amplitude(self, elem_x: float) -> typing.Tuple[float, float]:
        """Get the scaling amplitude, assuming the element is right on the top surface."""

        hole_idx_lower = bisect.bisect_right(self._hole_cent_sorted, elem_x)
        hole_idx_upper = hole_idx_lower+1

        hole_dist_lower = abs(elem_x - self._hole_cent_sorted[hole_idx_lower])
        hole_dist_upper = abs(elem_x - self._hole_cent_sorted[hole_idx_upper])

        if hole_dist_lower < hole_dist_upper:
            closest_idx = hole_idx_lower

        else:
            closest_idx = hole_idx_upper

        closest_hole = self._hole_cent_sorted[closest_idx]
        return closest_hole, self._hole_cent_to_amp[closest_hole]

    def _scale_factor_one_elem(self, cent: st7.Vector3):
        x, _, _ = cent[:]

        closest_hole, hole_raw_amp = self._get_hole_and_amplitude(x)
        real_amplitude = self.get_element_surface_depth_ratio(cent) * hole_raw_amp

        hole_centre_distance = abs(x - closest_hole)
        in_hole = hole_centre_distance*2 < self._hole_width

        if in_hole:
            return 1.0 + real_amplitude

        else:
            return 1.0


    def _get_hole_x_cent_and_amplitude(self):
        LONG_ENOUGH = 1000
        hole_to_amp = dict()


        def add_hole(x):
            hole_to_amp[x] = random.uniform(self._amplitude - self._max_variation, self._amplitude + self._max_variation)

        add_hole(0.0)
        current_x = self._spacing
        while current_x < LONG_ENOUGH:
            add_hole(current_x)
            add_hole(-1*current_x)

            current_x += self._spacing

        return hole_to_amp

    def __str__(self):
        return f"{self.__class__.__name__}(spacing={self._spacing}, amplitude={self._amplitude}, y_depth={self._y_depth}, hole_width={self._hole_width}, max_variation={self._max_variation})"


class SingleHoleCentre(CentroidAwareScaling):
    """One hole, in the middle."""

    _hole_width: float

    def __init__(self, pt: parameter_trend.ParameterTrend, y_depth: float, amplitude: float, hole_width: float):
        self._elem_scale_fact = {}
        self._y_depth = y_depth
        self._amplitude = amplitude
        self._hole_width = hole_width
        self._parameter_trend = pt

    def _scale_factor_one_elem(self, cent: st7.Vector3):
        x, _, _ = cent[:]

        real_amplitude = self.get_element_surface_depth_ratio(cent) * self._amplitude

        in_hole = 2 * abs(x - self._x_cent) < self._hole_width

        if in_hole:
            return 1.0 + real_amplitude

        else:
            return 1.0

    def __str__(self):
        return f"{self.__class__.__name__}(y_depth={self._y_depth}, amplitude={self._amplitude}, hole_width={self._hole_width}, )"
