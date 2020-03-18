import abc
import bisect
import collections
import math
import typing
import time

import numpy
import scipy.spatial

import st7
from common_types import T_Elem, T_Result


class Averaging:
    @abc.abstractmethod
    def populate_radius(
            self,
            node_positions: typing.Dict[int, st7.Vector3],
            element_connections: typing.Dict[T_Elem, typing.Tuple[int, ...]]
    ):
        """Called once at the start."""
        raise NotImplementedError

    @abc.abstractmethod
    def average_results(self, unaveraged: typing.Dict[T_Elem, T_Result]) -> typing.Dict[T_Elem, T_Result]:
        """Called at every iteration."""
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}()"


class NoAve(Averaging):
    def __init__(self):
        pass

    def populate_radius(
            self,
            node_positions: typing.Dict[int, st7.Vector3],
            element_connections: typing.Dict[T_Elem, typing.Tuple[int, ...]]
    ):
        pass

    def average_results(self, unaveraged: typing.Dict[T_Elem, T_Result]) -> typing.Dict[T_Elem, T_Result]:
        return unaveraged.copy()


class TreeData(typing.NamedTuple):
    id_to_pos: typing.Dict[int, st7.Vector3]
    idx_to_id: typing.Dict[int, int]
    np_data: numpy.array
    kd_tree: scipy.spatial.cKDTree


class AveInRadius(Averaging):
    _radius: float
    _element_connections: typing.Dict[T_Elem, typing.Tuple[int, ...]]
    _elem_nodal_contributions: typing.Dict[T_Elem, dict]  # The value in this is the relative contributions of all the nodes

    def __init__(self, radius: float):
        self._radius = radius

    def __str__(self):
        return f"{self.__class__.__name__}({self._radius})"

    def _populate_radius_bisect(
            self,
            node_positions: typing.Dict[int, st7.Vector3],
    ):

        relevant_ordinates = range(2)

        # Only consider the nodes which are connected to the elements in question (to avoid contact element, for example).
        all_connected_nodes = {iNode for connection in self._element_connections.values() for iNode in connection}
        node_positions_relevant = {iNode: xyz for iNode, xyz in node_positions.items() if iNode in all_connected_nodes}

        # Get the element centroids
        def get_cent(elem_conn):
            all_coords = [node_positions_relevant[iNode] for iNode in elem_conn]
            return sum(all_coords) / len(all_coords)

        elem_cents = {elem_id: get_cent(elem_conn) for elem_id, elem_conn in self._element_connections.items()}

        # Sort the nodes by x, y and z so we can find the candidates quickly(ish).
        idx_to_sorted_list = dict()
        for idx in relevant_ordinates:
            node_num_to_ordinate = {node_num: node_pos[idx] for node_num, node_pos in node_positions_relevant.items()}
            idx_to_sorted_list[idx] = sorted(
                (ordinate, node_num) for node_num, ordinate in node_num_to_ordinate.items())

        def candidate_nodes(elem_cent):
            """Get the candidate node numbers which could be close to a given element (but may not be within the radius)"""

            def nodes_with_range_of(idx):
                lower_val = elem_cent[idx] - self._radius
                upper_val = elem_cent[idx] + self._radius

                lower_idx = bisect.bisect_left(idx_to_sorted_list[idx], (lower_val, math.inf))
                upper_idx = bisect.bisect_right(idx_to_sorted_list[idx], (upper_val, 0))

                ordinates_and_node_nums = idx_to_sorted_list[idx][lower_idx:upper_idx]
                return {node_num for _, node_num in ordinates_and_node_nums}

            set_list = [nodes_with_range_of(idx) for idx in relevant_ordinates]
            all_candidate_nodes = set.intersection(*set_list)
            return all_candidate_nodes

        # Get the contributions of each node to each element.
        elem_nodal_contributions_raw = collections.defaultdict(dict)
        for elem_id, elem_xyz in elem_cents.items():
            for iNode in candidate_nodes(elem_xyz):
                dist = abs(node_positions_relevant[iNode] - elem_xyz)
                if dist < self._radius:
                    elem_nodal_contributions_raw[elem_id][iNode] = self._radius - dist

        # Normalise the contributions so they sum to one.
        self._elem_nodal_contributions = dict()
        for elem_id, contrib_dict in elem_nodal_contributions_raw.items():
            total = sum(contrib_dict.values())
            self._elem_nodal_contributions[elem_id] = {iNode: raw / total for iNode, raw in contrib_dict.items()}


    def _get_tree_data(self, id_to_pos: typing.Dict[int, st7.Vector3]) -> TreeData:

        indexed_id_to_pos = [(idx, iNode, xyz) for idx, (iNode, xyz) in enumerate(sorted(id_to_pos.items()))]
        idx_to_id = {idx: iNode for idx, iNode, _ in indexed_id_to_pos}

        np_data = numpy.zeros((len(indexed_id_to_pos), 2))
        for idx, iNode, xyz in indexed_id_to_pos:
            np_data[idx, :] = xyz[0:2]
            if xyz[2] != 0.0:
                raise ValueError("Expected 2D!")

        kd_tree = scipy.spatial.cKDTree(np_data)

        return TreeData(
            id_to_pos=id_to_pos,
            idx_to_id=idx_to_id,
            np_data=np_data,
            kd_tree=kd_tree,
        )

    def _populate_radius_cKDTree(self, node_positions: typing.Dict[int, st7.Vector3]):
        # Node tree
        nodes = self._get_tree_data(node_positions)

        # Element centroid tree
        all_connected_nodes = {iNode for connection in self._element_connections.values() for iNode in connection}
        node_positions_relevant = {iNode: xyz for iNode, xyz in node_positions.items() if iNode in all_connected_nodes}

        def get_cent(elem_conn):
            all_coords = [node_positions_relevant[iNode] for iNode in elem_conn]
            return sum(all_coords) / len(all_coords)

        elem_cents = {elem_id: get_cent(elem_conn) for elem_id, elem_conn in self._element_connections.items()}

        elems = self._get_tree_data(elem_cents)

        # Get the contributions of each node to each element.
        elem_nodal_contributions_raw = collections.defaultdict(dict)

        result = elems.kd_tree.query_ball_tree(nodes.kd_tree, self._radius)
        for elem_idx, list_of_node_idxs in enumerate(result):
            elem_id = elems.idx_to_id[elem_idx]
            elem_xyz = elems.id_to_pos[elem_id]
            node_nums = [nodes.idx_to_id[node_idx] for node_idx in list_of_node_idxs]

            elem_slice = elems.np_data[[elem_idx], :]
            node_slice = nodes.np_data[list_of_node_idxs, :]

            distance_matrix_local = scipy.spatial.distance_matrix(elem_slice, node_slice)

            for node_slice_idx, iNode in enumerate(node_nums):
                dist = distance_matrix_local[0, node_slice_idx]
                if dist < self._radius:
                    elem_nodal_contributions_raw[elem_id][iNode] = self._radius - dist


        # Normalise the contributions so they sum to one.
        self._elem_nodal_contributions = dict()
        for elem_id, contrib_dict in elem_nodal_contributions_raw.items():
            total = sum(contrib_dict.values())
            self._elem_nodal_contributions[elem_id] = {iNode: raw / total for iNode, raw in contrib_dict.items()}


    def populate_radius(
            self,
            node_positions: typing.Dict[int, st7.Vector3],
            element_connections: typing.Dict[T_Elem, typing.Tuple[int, ...]]
    ):

        t_start = time.time()

        self._element_connections = element_connections

        self._populate_radius_cKDTree(node_positions)
        # self._populate_radius_bisect(node_positions)

        # If there are no nodal contributions to an element, that will cause problems later on.
        missing_elements = [elem_id for elem_id in self._element_connections if elem_id not in self._elem_nodal_contributions]
        if missing_elements:
            raise ValueError(f"Missing {len(missing_elements)} elements... {missing_elements[0:3]}")

        t_end = time.time()
        print(f"Took {t_end-t_start:.4g} seconds to calculate averaging contributions with {len(self._element_connections)} elements at radius of {self._radius}.")

    def _nodal_to_elem(self, nodal_results, elem_id: T_Elem) -> T_Result:
        these_contribs = self._elem_nodal_contributions[elem_id]
        components = [factor * nodal_results[iNode] for iNode, factor in these_contribs.items()]
        return sum(components)

    def average_results(self, unaveraged: typing.Dict[T_Elem, T_Result]) -> typing.Dict[T_Elem, T_Result]:
        # Accumulate the nodal contributions of the elements.
        nodal_result_list = collections.defaultdict(list)
        for elem_id, result in unaveraged.items():
            for iNode in self._element_connections[elem_id]:
                nodal_result_list[iNode].append(result)

        nodal_results = {iNode: sum(results)/len(results) for iNode, results in nodal_result_list.items()}

        # Distribute the nodal results to the elements according to the pre-computed radius functions.
        averaged_results = {elem_id: self._nodal_to_elem(nodal_results, elem_id) for elem_id in unaveraged}
        return averaged_results