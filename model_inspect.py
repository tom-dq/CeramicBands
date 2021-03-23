
import bisect
import collections
import typing

import st7


def get_boundary_nodes(model: st7.St7Model) -> typing.FrozenSet[int]:
    """See how many plates each node is connected to as a non-generalisable way of getting the boundary."""

    elem_count = collections.Counter()

    for plate_num in model.entity_numbers(st7.Entity.tyPLATE):
        nodes = model.St7GetElementConnection(st7.Entity.tyPLATE, plate_num)
        for node in nodes:
            elem_count[node] += 1

    # One is corner and two is an edge.
    return frozenset(iNode for iNode, plate_count in elem_count.items() if plate_count < 3)


def get_element_columns(elem_centroid: typing.Dict[int, st7.Vector3], elem_volume: typing.Dict[int, float]
                        ) -> typing.Dict[float, typing.FrozenSet[int]]:

    columns = collections.defaultdict(set)

    elem_volume_cutoff = 1.25 * min(elem_volume.values())
    small_elems = {elem for elem, vol in elem_volume.items() if vol <= elem_volume_cutoff}
    for elem in small_elems:
        x_val = round(elem_centroid[elem].x, 12)
        columns[x_val].add(elem)

    return {x_val: frozenset(elem_nums) for x_val, elem_nums in columns.items()}
