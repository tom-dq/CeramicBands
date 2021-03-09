import numpy
import math

def make_D(E, rho) -> numpy.array:
    """Makes the full 6x6 elasticity matrix from sigma = D @ strain"""

    fact = E / (1+rho) / (1-2*rho)
    shear_diag = (1 - 2*rho) / 2

    return fact * numpy.array([
        [1-rho, rho, rho, 0, 0, 0],
        [rho, 1-rho, rho, 0, 0, 0],
        [rho, rho, 1-rho, 0, 0, 0],
        [0, 0, 0, shear_diag, 0, 0],
        [0, 0, 0, 0, shear_diag, 0],
        [0, 0, 0, 0, 0, shear_diag],
    ])



# Element depth to strain-recovery-factor estimate.
# For a if an element is totally free, a pre-strain will be completely converted into deformational strain.
# If an element is totally restrained, there can be no deformational strain.
# This estimates the ratio of output deformational strain based on the number of elements deep we are.
layer_and_ratio_x = [
    (0, -0.739749185053951),
    (1, -0.522574586045542),
    (2, -0.502164469093069),
    (3, -0.498797970628272),
    (4, -0.497531949117367),
    (5, -0.496896410310047),
    (6, -0.49652955470397),
    (7, -0.496298464554842),
    (8, -0.496143528540327),
    (9, -0.496034609925053),
    (10, -0.495955125291513),
    (11, -0.495895336825825),
    (12, -0.495849222287182),
    (13, -0.495812895840062),
]

layer_and_ratio_y = [
    (0, -0.596347298567013),
    (1, -0.535327136499151),
    (2, -0.512048872536697),
    (3, -0.504538464396896),
    (4, -0.501167003739315),
    (5, -0.499382823949796),
    (6, -0.498330112327591),
    (7, -0.497659012422582),
    (8, -0.497205723799128),
    (9, -0.496885525417282),
    (10, -0.496651117695436),
    (11, -0.496474445173674),
    (12, -0.496338028375621),
    (13, -0.496230524423967),
]

def strain_recovery_from_elem_depth(axis: int, x: int) -> float:
    """Estimate the deformational strain you would get from 100% pre-strain in x, for a given length. Based on a curve fit."""

    if axis == 0:
        # x direction
        A, B, C = -0.2430418, 2.21461174, -0.49667137

    elif axis == 1:
        # y axis
        A, B, C = -0.0988267, 0.9343151, -0.4972724

    else:
        raise ValueError(axis)

    return A * math.exp(-B * x) + C



if __name__ == "__main__":
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt


    def func(x, a, b, c):
        return a * numpy.exp(-b * x) + c


    xdata = numpy.array([xy[0] for xy in layer_and_ratio_y])
    ydata = numpy.array([xy[1] for xy in layer_and_ratio_y])

    popt, pcov = curve_fit(func, xdata, ydata)

    print(popt)
    print(pcov)
    plt.plot(xdata, ydata, 'bo', label='data')
    #plt.plot(xdata, func(xdata, *popt), 'r-', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

    xx = numpy.linspace(0, 13, 1000)
    yy = [strain_recovery_from_elem_depth(1, x) for x in xx]
    plt.plot(xx, yy, 'k--')

    plt.legend()

    plt.show()



