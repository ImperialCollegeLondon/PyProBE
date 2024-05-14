"""Tests for the Simple_OCV_fit module."""
import numpy as np

from pybatdata.methods.ocv_fitting.Simple_OCV_fit import Simple_OCV_fit


def test_fit_ocv():
    """Test the fit_ocv method."""

    def _f_pe(z):
        return z**2 + 3

    def _f_ne(z):
        return np.log(-0.3 * z + 1) + 5 * np.sin(z)

    n_points = 1000
    capacity = np.linspace(0, 1, n_points)
    z_real = [0.2, 0.9, 0.1, 0.6]
    z_pe_real = np.linspace(z_real[0], z_real[1], n_points)
    z_ne_real = np.linspace(z_real[2], z_real[3], n_points)
    voltage = _f_pe(z_pe_real) - _f_ne(z_ne_real)

    z = np.linspace(0, 1, n_points)
    pe_data = _f_pe(z)
    ne_data = _f_ne(z)

    ne_data = np.vstack((z, ne_data)).T
    pe_data = np.vstack((z, pe_data)).T
    z_guess = [0.4, 0.8, 0.2, 0.6]
    cathode_limits, anode_limits, _, _, _, _ = Simple_OCV_fit.fit_ocv(
        capacity, voltage, ne_data, pe_data, z_guess
    )

    np.testing.assert_allclose(cathode_limits, [z_real[:2]], rtol=1e-5)
    np.testing.assert_allclose(anode_limits, [z_real[2:]], rtol=1e-5)
