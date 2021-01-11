import numpy
from scipy.spatial import distance_matrix
import typing
import random
import st7
import math


class OrientationDistribution(typing.NamedTuple):
    num_seeds: int
    n_exponent: float



def distribute_naive(
    elem_centroid: typing.Dict[int, st7.Vector3],
    seed_points: typing.Dict[st7.Vector3, float],
    n: float,
):
    """Distributes the seed points according to the norm of how close they are to the centroids"""

    elem_values = dict()

    for iElem, elem_xyz in elem_centroid.items():
        dist_and_val = []
        for seed_xyz, seed_val in seed_points.items():
            dist = abs(seed_xyz - elem_xyz)
            dist_and_val.append(
                (dist, seed_val)
            )

        numerator = sum(1/dist**n * seed_val for dist, seed_val in dist_and_val)
        den = sum(1/dist**n for dist, _ in dist_and_val)

        elem_values[iElem] = numerator / den

    return elem_values



def distribute(
    elem_centroid: typing.Dict[int, st7.Vector3],
    seed_points: typing.Dict[st7.Vector3, float],
    n: float,
):

    # Enumerate the points going into the distance matrix
    elem_cent_order = {e_idx: (e_num, e_xyz) for e_idx, (e_num, e_xyz) in enumerate(elem_centroid.items()) }
    seed_order = {s_idx: (s_xyz, s_val) for s_idx, (s_xyz, s_val) in enumerate(seed_points.items()) }

    elem_cent_dense = [e_xyz for _, e_xyz in elem_cent_order.values()]
    seed_xyz_dense = [s_xyz for s_xyz, _ in seed_order.values()]

    # Get the distance matrix
    distances = distance_matrix(elem_cent_dense, seed_xyz_dense, p=2)

    # Make the "seed value" matrix of the same shape
    seed_val_mat = numpy.ones_like(distances)
    for s_idx, (_, s_val) in seed_order.items():
        seed_val_mat[:, s_idx] = s_val

    dist_weight_mat = 1 / distances ** n
    
    # Calculate the overall contribution
    num = numpy.sum(seed_val_mat * dist_weight_mat, axis=1)
    den = numpy.sum(dist_weight_mat, axis=1)

    elem_val_dense = num / den
    
    # Back to original format
    elem_values = dict()
    for e_idx, (e_num, _) in elem_cent_order.items():
        elem_values[e_num] = float(elem_val_dense[e_idx])

    return elem_values
    

def random_angle_distribution_360deg(
    dist_params: OrientationDistribution,
    elem_centroid: typing.Dict[int, st7.Vector3],
):
    """Make a random distribution of angles for each element"""

    x_points = [xyz.x for xyz in elem_centroid.values()]
    y_points = [xyz.y for xyz in elem_centroid.values()]

    def get_range(vals):
        val_min = min(vals)
        val_max = max(vals)
        val_range = val_max - val_min

        # Have a bit of overflow so there's no edge effect.
        buffer = 0.1 * val_range
        return val_min - buffer, val_max + buffer

    x_min, x_max = get_range(x_points)
    y_min, y_max = get_range(y_points)

    def make_seed_centroid():
        return st7.Vector3(random.uniform(x_min, x_max), random.uniform(y_min, y_max), 0.0)

    seed_xyzs = [make_seed_centroid() for _ in range(dist_params.num_seeds)]

    x_orient_vectors = {s_xyz: random.uniform(-1.0, 1.0) for s_xyz in seed_xyzs}
    y_orient_vectors = {s_xyz: random.uniform(-1.0, 1.0) for s_xyz in seed_xyzs}

    # Get the orientation vector at each of the points, then turn it into an angle
    x_e = distribute(elem_centroid, x_orient_vectors, dist_params.n_exponent)
    y_e = distribute(elem_centroid, y_orient_vectors, dist_params.n_exponent)

    elem_to_angle = {
        elem_num: math.degrees(math.atan2(y_e[elem_num], x_e[elem_num])) for elem_num in elem_centroid
    }

    return elem_to_angle




if __name__ == "__main__":
    elem_centroids = {
        1: st7.Vector3(1.0, 1.0, 0.0),
        2: st7.Vector3(1.0, 2.0, 0.0),
        3: st7.Vector3(1.0, 3.0, 0.0),
        4: st7.Vector3(1.0, 4.0, 0.0),
        5: st7.Vector3(2.0, 1.0, 0.0),
        6: st7.Vector3(2.0, 2.0, 0.0),
        7: st7.Vector3(2.0, 3.0, 0.0),
        8: st7.Vector3(2.0, 4.0, 0.0),
    }

    seed_points = {
        st7.Vector3(1.1, 1.1, 0.0): 10.0,
        st7.Vector3(1.5, 4.5, 0.0): 1.0,
    }


    N = 1.0
    elem_values_naive = distribute_naive(elem_centroids, seed_points, N)
    elem_values = distribute(elem_centroids, seed_points, N)

    if elem_values_naive != elem_values:
        print(N)
        print(elem_values_naive)
        print(elem_values)


