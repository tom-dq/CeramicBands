import abc
import typing
import math

import st7



class BasePerturbation:
    x_cent: float
    y_high: float
    y_low: float

    @abc.abstractmethod
    def top_surface_dip(self, x: float) -> float:
        pass

    def update_node_locations(self, node_xyz: typing.Dict[int, st7.Vector3]) -> typing.Dict[int, st7.Vector3]:
        x_vals = {xyz.x for xyz in node_xyz.values()}
        y_vals = {xyz.y for xyz in node_xyz.values()}

        self.x_cent = 0.5 * (min(x_vals) + max(x_vals))

        self.y_low = min(y_vals)
        self.y_high = max(y_vals)

        moved_nodes = {}
        for iNode, xyz in node_xyz.items():
            dip_top_surface = self.top_surface_dip(xyz.x)
            # Full movement at the top, down to nothing at the bottom.

            depth_ratio = (xyz.y - self.y_low) / (self.y_high - self.y_low)
            y_dip = dip_top_surface * depth_ratio

            moved_nodes[iNode] = xyz._replace(y=xyz.y - y_dip)

        return moved_nodes


class NoPerturb(BasePerturbation):
    def __str__(self):
        return "NoPerturb()"

    def top_surface_dip(self, x: float) -> float:
        return 0.0


class IndentCentre(BasePerturbation):

    def __init__(self, angle_deg: float, depth: float):
        self.angle_deg = angle_deg
        self.depth = depth

    def __str__(self):
        return f"IndentCentre({self.angle_deg}, {self.depth})"

    def top_surface_dip(self, x: float) -> float:

        x_flat = self.depth / math.tan(math.radians(self.angle_deg))

        x_from_cent = abs(self.x_cent - x)
        if x_from_cent < x_flat:
            dip_ratio = (x_flat - x_from_cent) / x_flat
            return self.depth * dip_ratio

        else:
            return 0.0



class SphericalIndentCenter(BasePerturbation):

    def __init__(self, radius: float, depth: float):
        self.radius = radius
        self.depth = depth

    def __str__(self):
        return f"SphericalIndentCenter({self.radius}, {self.depth})"

    def top_surface_dip(self, x: float) -> float:

        delta_x = abs(x - self.x_cent)

        try:
            delta_y = math.sqrt(self.radius**2 - delta_x**2)

        except ValueError:
            # Not within the range of the circle
            return 0.0

        circle_cent_y = self.y_high - self.depth + self.radius
        y_on_circle = circle_cent_y - delta_y

        y_dip = self.y_high - y_on_circle

        return max(y_dip, 0.0)



