"""Tests for the convert.py module."""

# pylint: disable=C0116

import numpy as np

from ms_zarr_converter import convert


def test_reshape_column():
    # Arange
    column_data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    cshape = (2, 4)
    time_indices = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    baselines = np.array([0, 1, 2, 3, 0, 1, 2, 3])

    reshaped_column_expected = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])

    # Act
    reshaped_column = convert.reshape_column(
        column_data,
        cshape,
        time_indices,
        baselines,
    )

    # Assert
    np.testing.assert_array_almost_equal(
        reshaped_column, reshaped_column_expected
    )
