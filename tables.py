"""Some bits and pieces to make working with Strand7 tables easier"""
import bisect
import math
import numbers

import typing

from common_types import XY


class Table:
    _data_set: bool = False
    data: typing.Tuple[XY] = None
    _data_x: typing.Tuple[float] = None
    max_abs_y: float

    """Keeps track of some xy data"""

    def __init__(self):
        self._data_set = False

    def set_table_data(self, data: typing.Sequence[XY]):
        """This is a method so the data can be adjusted without loosing the reference to the object."""

        # Fail if it's not sorted, or if any compare equal.
        tuple_data = tuple(data)
        if not all(tuple_data[i].x < tuple_data[i+1].x for i in range(len(tuple_data)-1)):
            print(tuple_data)
            raise ValueError("Expecting the data to be sorted and unique.")

        self.data = tuple_data
        self._data_x = tuple(xy.x for xy in self.data)
        self.max_abs_y = max(abs(xy.y) for xy in self.data)

        self._data_set = True

    def copy_scaled(self, x_scale: numbers.Number, y_scale: numbers.Number) -> "Table":
        new_table = Table()

        scaled_data = (XY(x=x_scale*d.x, y=y_scale*d.y) for d in self.data)
        new_table.set_table_data(scaled_data)

        return new_table


    def interp(self, x: float) -> float:

        if not self._data_set:
            raise ValueError("Need to call .set_table_data(data) first.")

        # If we're off the botton or top, just return the final value.
        if x <= self._data_x[0]:
            return self.data[0].y

        elif self._data_x[-1] <= x:
            return self.data[-1].y

        # Actually have to look up / interpolate.
        index_lower_or_equal = bisect.bisect_right(self._data_x, x) - 1

        # Off the end.
        if index_lower_or_equal == len(self.data)-1:
            return self.data[-1].y

        low = self.data[index_lower_or_equal]

        high = self.data[index_lower_or_equal + 1]

        # If we're right on the point, return it!
        if math.isclose(x, low.x):
            return low.y

        x_range = high.x - low.x
        y_range = high.y - low.y
        grad = y_range / x_range
        delta_x = x - low.x

        return low.y + delta_x * grad

    def min_val(self) -> float:

        if not self._data_set:
            raise ValueError("Need to call .set_table_data(data) first.")

        return self.data[0].y

    def with_appended_datapoint(self, xy: XY) -> "Table":
        """Build the next bit of a step-wise sequence."""

        if not self._data_set:
            raise ValueError("Need to call .set_table_data(data) first.")

        # TODO - this could certainly be made more robust...

        EPS = 1e-6

        existing_final = self.data[-1]
        is_same_x = math.isclose(xy.x, existing_final.x)
        is_same_y = math.isclose(xy.y, existing_final.y)
        if is_same_y:
            if is_same_x:
                # Don't need to do anything!
                new_data = self.data

            else:
                new_data = self.data + (xy,)

        else:
            new_points = []

            need_to_extend_old_data = existing_final.x < (xy.x - EPS)

            if need_to_extend_old_data:
                new_points.append(existing_final._replace(x=xy.x - EPS))

            new_points.append(xy._replace(x=xy.x + EPS))
            new_data = self.data + tuple(new_points)

        working_table = Table()
        working_table.set_table_data(new_data)
        return working_table

    def as_flat_doubles(self) -> typing.Sequence[float]:
        """To go into the Strand7 API, with everything flattened."""

        if not self._data_set:
            raise ValueError("Need to call .set_table_data(data) first.")

        for (x, y) in self.data:
            yield x
            yield y


def test_simple_table():

    t_transition1 = 0.1
    t_transition2 = 0.4

    y1 = 34.5
    y2 = 21.1

    t_init = Table([
        XY(0.0, 0.0),
        XY(t_transition1, 0.0)
    ])

    t_int1 = t_init.with_appended_datapoint(XY(t_transition1, y1))
    t_int2 = t_int1.with_appended_datapoint(XY(t_transition2, y2))

    print(t_int2.data)

if __name__ == "__main__":
    test_simple_table()
