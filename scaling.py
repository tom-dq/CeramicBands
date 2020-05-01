import abc
import math
import random
import typing

import st7
from common_types import T_ScaleKey


class Scaling:
    """Scales the "Yield Stress" according to some criterion."""
    @abc.abstractmethod
    def get_x_scale_factor(self, scale_key: T_ScaleKey) -> float:
        raise NotImplementedError()

    def __str__(self):
        return self.__class__.__name__

    def assign_centroids(self, elem_centroid: typing.Dict[int, st7.Vector3]):
        """This is not used by all the methods."""
        pass


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
        if scale_key not in self._x_scaler:
            self._x_scaler[scale_key] = 1.0 + random.uniform(-1 * self._random_spread, self._random_spread)

        return self._x_scaler[scale_key]

    def __str__(self):
        return f"{self.__class__.__name__}(random_spread={self._random_spread})"


class CentroidAwareScaling(Scaling):
    """Keeps track of where the elements are."""

    _elem_scale_fact: dict
    _spacing: float
    _amplitude: float
    _y_min: float
    _y_max: float
    _y_depth: float
    _x_cent: float

    @abc.abstractmethod
    def _scale_factor_one_elem(self, cent: st7.Vector3):
        raise NotImplementedError()

    def assign_centroids(self, elem_centroid: typing.Dict[int, st7.Vector3]):
        # Find out how deep the elements go.
        self._y_max = max(xyz.y for xyz in elem_centroid.values())
        self._y_min = self._y_max - self._y_depth
        self._x_cent = 0.5 * (min(xyz.x for xyz in elem_centroid.values()) + max(xyz.x for xyz in elem_centroid.values()))

        for elem_num, elem_cent in elem_centroid.items():
            self._elem_scale_fact[elem_num] = self._scale_factor_one_elem(elem_cent)

    def get_x_scale_factor(self, scale_key: T_ScaleKey) -> float:
        elem_num, _ = scale_key[:]
        return self._elem_scale_fact[elem_num]


class CosineScaling(CentroidAwareScaling):
    """Does a cosine based on the element position."""

    def __init__(self, y_depth: float, spacing: float, amplitude: float):
        self._elem_scale_fact = dict()
        self._y_depth = y_depth
        self._spacing = spacing
        self._amplitude = amplitude

    def _scale_factor_one_elem(self, cent: st7.Vector3):
        x, y, _ = cent[:]
        if y < self._y_min:
            real_amplitude = 0.0

        elif y < self._y_max:
            real_amplitude = self._amplitude * (y-self._y_min) / (self._y_max - self._y_min)

        else:
            real_amplitude = self._amplitude

        a = self._spacing / (math.pi * 2)
        raw_cosine = math.cos(a * x)

        return 1.0 + real_amplitude * raw_cosine

    def __str__(self):
        return f"{self.__class__.__name__}(spacing={self._spacing}, amplitude={self._amplitude}, y_depth={self._y_depth})"


class SpacedStepScaling(CentroidAwareScaling):
    """Mimic the regular spacing of the holes drilled in the surface."""

    _hole_width: float

    def __init__(self, y_depth: float, spacing: float, amplitude: float, hole_width: float):
        self._elem_scale_fact = dict()
        self._y_depth = y_depth
        self._spacing = spacing
        self._amplitude = amplitude
        self._hole_width = hole_width

    def _scale_factor_one_elem(self, cent: st7.Vector3):
        x, y, _ = cent[:]

        if y < self._y_min:
            real_amplitude = 0.0

        elif y < self._y_max:
            real_amplitude = self._amplitude * (y-self._y_min) / (self._y_max - self._y_min)

        else:
            real_amplitude = self._amplitude

        # See how close we are to a "hole"
        closest_hole = 0.0
        working_hole_distance = abs(x - closest_hole)
        while working_hole_distance > 0.5*self._spacing:
            if x > closest_hole:
                closest_hole += self._spacing

            elif x < closest_hole:
                closest_hole -= self._spacing

            working_hole_distance = abs(x - closest_hole)

        hole_centre_distance = abs(x - closest_hole)
        in_hole = hole_centre_distance*2 < self._hole_width

        if in_hole:
            return 1.0 + real_amplitude

        else:
            return 1.0

    def __str__(self):
        return f"{self.__class__.__name__}(spacing={self._spacing}, amplitude={self._amplitude}, y_depth={self._y_depth}, hole_width={self._hole_width})"


class SingleHoleCentre(CentroidAwareScaling):
    """One hole, in the middle."""

    _hole_width: float

    def __init__(self, y_depth: float, amplitude: float, hole_width: float):
        self._elem_scale_fact = dict()
        self._y_depth = y_depth
        self._amplitude = amplitude
        self._hole_width = hole_width

    def _scale_factor_one_elem(self, cent: st7.Vector3):
        x, y, _ = cent[:]
        if y < self._y_min:
            real_amplitude = 0.0

        elif y < self._y_max:
            real_amplitude = self._amplitude * (y-self._y_min) / (self._y_max - self._y_min)

        else:
            real_amplitude = self._amplitude


        in_hole = 2 * abs(x - self._x_cent) < self._hole_width

        if in_hole:
            return 1.0 + real_amplitude

        else:
            return 1.0


    def __str__(self):
        return f"{self.__class__.__name__}(y_depth={self._y_depth}, amplitude={self._amplitude}, hole_width={self._hole_width}, )"
