"""Tests for the Simple_OCV_fit module."""
import numpy as np

from pyprobe.methods.ocv_fitting.Simple_OCV_fit import Simple_OCV_fit


def graphite_LGM50_ocp_Chen2020(sto):
    """Chen2020 graphite ocp fit."""
    u_eq = (
        1.9793 * np.exp(-39.3631 * sto)
        + 0.2482
        - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
        - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
    )

    return u_eq


def nmc_LGM50_ocp_Chen2020(sto):
    """Chen2020 nmc ocp fit."""
    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )

    return u_eq


def test_fit_ocv():
    """Test the fit_ocv method."""
    n_points = 1000
    capacity = np.linspace(0, 1, n_points)
    x_real = [0.2, 0.9, 0.1, 0.7]
    x_pe_real = np.linspace(x_real[0], x_real[1], n_points)
    x_ne_real = np.linspace(x_real[2], x_real[3], n_points)
    voltage = nmc_LGM50_ocp_Chen2020(1 - x_pe_real) - graphite_LGM50_ocp_Chen2020(
        x_ne_real
    )

    z = np.linspace(0, 1, n_points)
    ocp_pe = nmc_LGM50_ocp_Chen2020(z)
    ocp_ne = graphite_LGM50_ocp_Chen2020(z)

    x_guess = [0.4, 0.8, 0.2, 0.6]
    x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi, _, _, _, _ = Simple_OCV_fit.fit_ocv(
        capacity, voltage, x_ne=z, ocp_ne=ocp_ne, x_pe=z, ocp_pe=ocp_pe, x_guess=x_guess
    )

    np.testing.assert_allclose(
        np.array([x_pe_lo, x_pe_hi, x_ne_lo, x_ne_hi]), np.array(x_real), rtol=1e-4
    )
