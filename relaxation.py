import abc
import typing

from common_types import T_ScaleKey, T_FactVal


# Relaxation controls some "backoff" when a big jump in dilation is encountered.
class Relaxation:
    _history: typing.Dict[T_ScaleKey, T_FactVal]
    _prev_factor: float = None
    _this_factor: float = None

    def set_current_relaxation(self, prev_factor: float, this_factor: float):
        self._prev_factor = prev_factor
        self._this_factor = this_factor

    @abc.abstractmethod
    def relaxed(self, scale_key, new_value) -> float:
        raise NotImplementedError


    def _lookup_key(self, scale_key, factor):
        return (scale_key, factor)


    def _get_prev_val(self, scale_key):
        # If this is the first round, there should be no previous value, so take it to be zero. Otherwise, should be something there!
        lookup_key = self._lookup_key(scale_key, self._prev_factor)
        if self._prev_factor == 0.0:
            if lookup_key in self._history:
                raise ValueError("Why is there something in history on the first step?")

            return 0.0

        else:
            return self._history[lookup_key]

    def _set_current_val(self, scale_key, new_val_relaxed):
        # Save the value for next time.
        self._history[self._lookup_key(scale_key, self._this_factor)] = new_val_relaxed

    def flush_previous_values(self):
        """Call this after doing the relaxation for a step - it clears detritus from the history."""
        keys_to_delete = [(scale_key, factor) for (scale_key, factor) in self._history.keys() if factor == self._prev_factor]
        for key in keys_to_delete:
            del self._history[key]


class ProportionalRelaxation(Relaxation):
    """Applies some relaxation to the ramp-up of the pre-strain, for example, limit it to 50% of the raw increase."""
    _incremental_ratio: float = None

    def __init__(self, incremental_ratio: float):
        self._incremental_ratio = incremental_ratio
        self._history = dict()

    def __str__(self):
        return f"{self.__class__.__name__}(incremental_ratio={self._incremental_ratio})"

    def relaxed(self, scale_key, new_value) -> float:
        """prev_factor and new_factor are the load case / freedom case factor.
           new_value is the "full" prestrain at the current step."""

        prev_val = self._get_prev_val(scale_key)

        delta_value_full = new_value - prev_val
        relaxed_value = prev_val + self._incremental_ratio * delta_value_full

        self._set_current_val(scale_key, relaxed_value)

        return relaxed_value


class LimitedIncreaseRelaxation(Relaxation):
    """Clip the new value if it's gone up to much."""
    _strain_change_limit: float = None

    def __init__(self, strain_change_limit: float):
        self._strain_change_limit = strain_change_limit
        self._history = dict()

    def __str__(self):
        return f"{self.__class__.__name__}(strain_change_limit={self._strain_change_limit})"

    def relaxed(self, scale_key, new_value) -> float:
        """prev_factor and new_factor are the load case / freedom case factor.
           new_value is the "full" prestrain at the current step."""

        prev_val = self._get_prev_val(scale_key)

        delta_value_full = new_value - prev_val

        if abs(delta_value_full) > self._strain_change_limit:
            # Need to clip it a bit.
            if delta_value_full > 0:
                relaxed_value = prev_val + self._strain_change_limit

            elif delta_value_full < 0:
                relaxed_value = prev_val - self._strain_change_limit

            else:
                raise ValueError("Zero?")

        else:
            relaxed_value = new_value

        self._set_current_val(scale_key, relaxed_value)

        return relaxed_value