import abc
import typing
import math

import st7



class BasePerturbation:
    
    @abc.abstractmethod
    def update_node_locations(self, node_xyz: typing.Dict[int, st7.Vector3]) -> typing.Dict[int, st7.Vector3]:
        pass


class NoPerturb(BasePerturbation):
    
    def update_node_locations(self, node_xyz: typing.Dict[int, st7.Vector3]) -> typing.Dict[int, st7.Vector3]:
        return node_xyz.copy()

    def __str__(self):
        return "NoPerturb()"

class IndentCentre(BasePerturbation):

    def __init__(self, angle_deg: float, depth: float):
        self.angle_deg = angle_deg
        self.depth = depth

    def __str__(self):
        return f"IndentCentre({self.angle_deg}, {self.depth})"


    def _dip_at(self, x_cent, x):

        x_flat = self.depth / math.tan(math.radians(self.angle_deg))

        x_from_cent = abs(x_cent - x)
        if x_from_cent < x_flat:
            dip_ratio = (x_flat - x_from_cent) / x_flat
            return self.depth * dip_ratio

        else:
            return 0.0

    def update_node_locations(self, node_xyz: typing.Dict[int, st7.Vector3]) -> typing.Dict[int, st7.Vector3]:
        
        x_vals = {xyz.x for xyz in node_xyz.values()}
        y_vals = {xyz.y for xyz in node_xyz.values()}

        x_cent = 0.5 * (min(x_vals) + max(x_vals))

        y_low = min(y_vals)
        y_high = max(y_vals)

        moved_nodes = {}
        for iNode, xyz in node_xyz.items():
            dip_top_surface = self._dip_at(x_cent, xyz.x)
            # Full movement at the top, down to nothing at the bottom.

            depth_ratio = (xyz.y - y_low) / (y_high - y_low)
            y_dip = dip_top_surface * depth_ratio

            moved_nodes[iNode] = xyz._replace(y=xyz.y-y_dip)

        return moved_nodes

