"""Main module for converting a measurement_set to a Zarr format."""

# pylint: disable=W0511,R0913,R0914,R0915

import multiprocessing as mp
import os
import shutil
from copy import deepcopy

import numba as nb
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from casacore import tables
from dask.distributed import Client, LocalCluster, wait

from ms_zarr_converter.constants import (
    COLUMN_DIMENSIONS,
    COLUMN_TO_DATA_VARIABLE_NAMES,
    DATA_VARIABLE_COLUMNS,
    STOKES_TYPES,
)


@nb.njit(parallel=False, fastmath=True)
def searchsorted_nb(array_1, array_2):
    """Compiled version of searchsort."""
    result_array = np.empty(len(array_2), np.intp)
    for i, value in enumerate(array_2):
        result_array[i] = np.searchsorted(array_1, value)
    return result_array


@nb.njit(parallel=False, fastmath=True)
def isin_nb(array_1: np.ndarray, array_2: np.ndarray):
    """Compiled version of isin."""
    shape = array_1.shape
    array_1 = array_1.ravel()
    array_length = len(array_1)
    result_array = np.full(array_length, False)
    set_b = set(array_2)
    for i in range(array_length):
        if array_1[i] in set_b:
            result_array[i] = True
    return result_array.reshape(shape)


@nb.njit(parallel=False, fastmath=True)
def antennas_to_indices(antenna1, antenna2):
    """Convert pairs of antenna indices into baseline indices."""
    all_baselines = np.empty(antenna1.size, dtype=np.int32)

    # Required to give expected ordering
    # 20000 may be too small for certain Low observations
    max_num_antenna_pairs = 20000

    for i, antenna1_i in enumerate(antenna1):
        all_baselines[i] = (
            (antenna1_i + antenna2[i]) * (antenna1_i + antenna2[i] + 1)
        ) // 2 + max_num_antenna_pairs * antenna1_i

    return all_baselines


@nb.njit(parallel=False, fastmath=True)
def invertible_indices(antenna1, antenna2):
    """Convert pairs of antenna indices into invertible baseline indices."""
    all_baselines = np.empty(antenna1.size, dtype=np.int32)

    for i, antenna1_i in enumerate(antenna1):
        all_baselines[i] = (
            (antenna1_i + antenna2[i]) * (antenna1_i + antenna2[i] + 1)
        ) // 2 + antenna2[i]

    return all_baselines


@nb.njit(parallel=False, fastmath=True)
def indices_to_baseline_ids(unique_baselines):
    """Convert unique baseline indices to pairs of baseline IDs."""
    baseline_id1 = np.empty(unique_baselines.size, dtype=np.int32)
    baseline_id2 = np.empty(unique_baselines.size, dtype=np.int32)

    for i, unique_baseline in enumerate(unique_baselines):
        width = (np.sqrt(8 * unique_baseline + 1) - 1) // 2
        triangular_number = (width * (width + 1)) / 2
        remainder = unique_baseline - triangular_number
        baseline_id2[i] = remainder
        baseline_id1[i] = width - remainder

    return baseline_id1, baseline_id2


def get_dir_size(path="."):
    """Gets the total directory size in bytes."""
    total = 0
    with os.scandir(path) as dir_entries:
        for entry in dir_entries:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


# Check that the time steps have a single unique value
def _check_interval_consistent(measurement_set):
    time_interval = pd.unique(measurement_set.getcol("INTERVAL"))
    assert len(time_interval) == 1, "Interval is not consistent."
    return time_interval[0]


# Check that the time steps have a single unique value
def _check_exposure_consistent(measurement_set):
    exposure_time = pd.unique(measurement_set.getcol("EXPOSURE"))
    assert len(exposure_time) == 1, "Exposure is not consistent."
    return exposure_time[0]


# Check that the FIELD_ID has a single unique value
def _check_single_field(measurement_set):
    field_id = pd.unique(measurement_set.getcol("FIELD_ID"))
    assert len(field_id) == 1, "More than one field present."
    return field_id[0]


def create_coordinates(
    xds,
    measurement_set,
    unique_times,
    baseline_ant1_id,
    baseline_ant2_id,
    fits_in_memory,
):
    """Create the DataArray coordinates and add various metadata."""
    # Field subtable
    field = measurement_set.FIELD[0]

    # Assumes only a single FIELD_ID
    delay_direction = {
        "dims": "",
        "data": field.get("DELAY_DIR", [0]).tolist()[0],
        "attrs": {
            "units": "rad",
            "type": "sky_coord",
            "description": "Direction of delay center in right ascension and \
                declination.",
        },
    }

    phase_direction = {
        "dims": "",
        "data": field.get("PHASE_DIR", [0]).tolist()[0],
        "attrs": {
            "units": "rad",
            "type": "sky_coord",
            "description": "Direction of phase center in right ascension and \
                declination.",
        },
    }

    reference_direction = {
        "dims": "",
        "data": field.get("REFERENCE_DIR", [0]).tolist()[0],
        "attrs": {
            "units": "rad",
            "type": "sky_coord",
            "description": "Direction of reference direction in right \
                ascension and declination. Used in single-dish to record the \
                    associated reference direction if position-switching has \
                        already been applied. For interferometric data, this \
                            is the original correlated field center, and may \
                                equal delay_direction or phase_direction.",
        },
    }

    field_info = {
        "name": field.get("NAME", "name"),
        "code": field.get("CODE", "1"),
        "field_id": _check_single_field(measurement_set),
    }

    xds.attrs["delay_direction"] = delay_direction
    xds.attrs["phase_direction"] = phase_direction
    xds.attrs["reference_direction"] = reference_direction
    xds.attrs["field_info"] = field_info

    coords = {
        "baseline_antenna1_id": ("baseline_id", baseline_ant1_id),
        "baseline_antenna2_id": ("baseline_id", baseline_ant2_id),
        "baseline_id": np.arange(len(baseline_ant1_id)),
    }

    # If it doesn't fit in memory, populate time coordinate on-the-fly
    if fits_in_memory:
        coords["time"] = unique_times

    # Add frequency coordinates
    frequencies = measurement_set.SPECTRAL_WINDOW[0].get("CHAN_FREQ", [])

    # Remove NaN values
    coords["frequency"] = frequencies[~np.isnan(frequencies)]

    # Add polatization coordinates
    polarizations = measurement_set.POLARIZATION[0].get("CORR_TYPE", [])
    coords["polarization"] = np.vectorize(STOKES_TYPES.get)(polarizations)

    # Add named coordinates
    xds = xds.assign_coords(coords)
    xds.frequency.attrs[
        "reference_frequency"
    ] = measurement_set.SPECTRAL_WINDOW[0].get("REF_FREQUENCY", "")
    xds.frequency.attrs[
        "effective_channel_width"
    ] = measurement_set.SPECTRAL_WINDOW[0].get("EFFECTIVE_BW", "")[0]

    channel_widths = measurement_set.SPECTRAL_WINDOW[0].get("CHAN_WIDTH", "")

    if not isinstance(channel_widths, str):
        unique_chan_width = np.unique(
            channel_widths[np.logical_not(np.isnan(channel_widths))]
        )
        xds.frequency.attrs["channel_width"] = {
            "data": np.abs(unique_chan_width[0]),
            "attrs": {"type": "quanta", "units": "Hz"},
        }

    return xds


def reshape_column(
    column_data,
    cshape,
    time_indices,
    baselines,
):
    """Reshapes a column."""
    # full data is the maximum of the data shape and chunk shape dimensions
    # for each time interval
    fulldata = np.full(
        cshape + column_data.shape[1:], np.nan, dtype=column_data.dtype
    )

    fulldata[time_indices, baselines] = column_data

    return fulldata


def get_baselines(measurement_set: tables.table) -> np.ndarray:
    """Gets all baseline pairs needed to reshape data columns"""
    # main table uses time x (antenna1,antenna2)
    antenna1, antenna2 = measurement_set.getcol(
        "ANTENNA1", 0, -1
    ), measurement_set.getcol("ANTENNA2", 0, -1)
    baselines = np.array(
        [
            str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
            for ll in np.unique(
                np.hstack([antenna1[:, None], antenna2[:, None]]), axis=0
            )
        ]
    )

    return baselines


def get_baseline_pairs(measurement_set: tables.table) -> tuple:
    """Gets the baseline antenna indices required for array broadcasting."""
    baselines = get_baselines(measurement_set)

    baseline_ant1_id, baseline_ant2_id = np.array(
        [tuple(map(int, x.split("_"))) for x in baselines]
    ).T

    return (baseline_ant1_id, baseline_ant2_id)


def get_baseline_indices(measurement_set: tables.table) -> tuple:
    """Calculate baseline indices for a given measurement set."""
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = measurement_set.getcol(
        "ANTENNA1", 0, -1
    ), measurement_set.getcol("ANTENNA2", 0, -1)

    all_antenna_pairs = antennas_to_indices(ant1, ant2)

    # pd.unique is much faster than np.unique because it doesn't pre-sort
    # If len(unique_antenna_pairs) << len(all_antenna_pairs) then this is
    # is a greater than 2x speedup
    unique_antenna_pairs = np.sort(pd.unique(all_antenna_pairs))

    # Compiled searchsort on pre-sorted arrays is ~2x faster
    baseline_indices = searchsorted_nb(unique_antenna_pairs, all_antenna_pairs)

    return baseline_indices


def get_invertible_indices(measurement_set: tables.table) -> tuple:
    """Calculate invertible baseline indices for a given measurement set."""
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = measurement_set.getcol(
        "ANTENNA1", 0, -1
    ), measurement_set.getcol("ANTENNA2", 0, -1)

    unique_invertible_indices = np.sort(
        pd.unique(invertible_indices(ant1, ant2))
    )

    return indices_to_baseline_ids(unique_invertible_indices)


def create_attribute_metadata(column_name, measurement_set):
    """Extract metadata for a given column in a measurement set."""
    attrs_metadata = {}

    if column_name in ["U", "V", "W"]:
        name = "UVW"
        column_description = measurement_set.getcoldesc(name)

        column_info = column_description.get("INFO", {})

        attrs_metadata["type"] = column_info.get("type", "None")
        attrs_metadata["ref"] = column_info.get("Ref", "None")

        attrs_metadata["units"] = column_description.get("keywords", {}).get(
            "QuantumUnits", ["None"]
        )[0]

    else:
        column_description = measurement_set.getcoldesc(column_name)

        column_info = column_description.get("INFO", {})

        attrs_metadata["type"] = column_info.get("type", "None")
        attrs_metadata["ref"] = column_info.get("Ref", "None")

        attrs_metadata["units"] = column_description.get("keywords", {}).get(
            "QuantumUnits", ["None"]
        )[0]

    return attrs_metadata


# Separate function to add time attributes. Must happen after the
# time coordinate is created
def add_time(xds, measurement_set):
    """Add time-related attributes to a Dataset based on a measurement set."""
    interval = _check_interval_consistent(measurement_set)
    exposure = _check_exposure_consistent(measurement_set)
    time_description = measurement_set.getcoldesc("TIME")

    xds.time.attrs["type"] = time_description.get("MEASINFO", {}).get(
        "type", "None"
    )

    xds.time.attrs["Ref"] = time_description.get("MEASINFO", {}).get(
        "Ref", "None"
    )

    xds.time.attrs["units"] = time_description.get("keywords", {}).get(
        "QuantumUnits", ["None"]
    )[0]

    xds.time.attrs["time_scale"] = (
        time_description.get("keywords", {})
        .get("MEASINFO", {})
        .get("Ref", "None")
    )

    xds.time.attrs["integration_time"] = interval

    xds.time.attrs["effective_integration_time"] = exposure


def ms_chunk_to_zarr(
    xds,
    infile,
    row_indices,
    times,
    num_unique_baselines,
    outfile,
    times_per_chunk,
    data_variable_columns,
    column_dimensions,
    column_to_data_variable_names,
):
    """Convert a measurement set chunk to a Zarr store in an xarray Dataset."""
    # TODO refactor MS loading
    # Loading+querying uses a lot of memory
    with tables.table(
        infile, readonly=True, lockoptions="autonoread"
    ).selectrows(row_indices) as measurement_set_chunk:
        # Get dimensions of data
        time_indices = searchsorted_nb(
            times, measurement_set_chunk.getcol("TIME")
        )
        data_shape = (len(times), num_unique_baselines)

        baseline_indices = get_baseline_indices(measurement_set_chunk)

        # Must loop over each column to create an xarray DataArray for each
        for column_name in data_variable_columns:
            column_data = measurement_set_chunk.getcol(column_name)

            # UVW column must be split into u, v, and w
            if column_name == "UVW":
                subcolumns = [
                    column_data[:, 0],
                    column_data[:, 1],
                    column_data[:, 2],
                ]
                subcolumn_names = ["U", "V", "W"]

                for data, name in zip(subcolumns, subcolumn_names):
                    reshaped_column = reshape_column(
                        data,
                        data_shape,
                        time_indices,
                        baseline_indices,
                    )

                    # Create a DataArray instead of appending immediately to
                    # Dataset so time coordinates can be updated
                    xda = xr.DataArray(
                        reshaped_column,
                        dims=column_dimensions.get(name),
                    ).assign_coords(time=("time", times))

                    # Add the DataArray to the Dataset
                    xds[column_to_data_variable_names.get(name)] = xda

            else:
                reshaped_column = reshape_column(
                    column_data,
                    data_shape,
                    time_indices,
                    baseline_indices,
                )

                # Create a DataArray instead of appending immediately to
                # Dataset so time coordinates can be updated
                xda = xr.DataArray(
                    reshaped_column,
                    dims=column_dimensions.get(column_name),
                ).assign_coords(time=("time", times))

                # Add the DataArray to the Dataset
                xds[column_to_data_variable_names.get(column_name)] = xda

                # Add column metadata at the end
                # Adding metadata to a variable means the variable must
                # already exist

                xds[column_to_data_variable_names[column_name]].attrs.update(
                    create_attribute_metadata(
                        column_name, measurement_set_chunk
                    )
                )

        xds = xds.chunk(
            {
                "time": times_per_chunk,
                "frequency": -1,
                "baseline_id": -1,
                "polarization": -1,
            }
        )

        add_time(xds, measurement_set_chunk)
        xds.to_zarr(store=outfile, mode="w", compute=True)


def ms_to_zarr_in_memory(
    xds,
    measurement_set,
    unique_times,
    num_unique_baselines,
    column_names,
    infile,
    outfile,
):
    """Convert a measurement set to a Zarr store in memory."""
    # Get dimensions of data
    time_indices = searchsorted_nb(
        unique_times, measurement_set.getcol("TIME")
    )
    data_shape = (len(unique_times), num_unique_baselines)

    baseline_indices = get_baseline_indices(measurement_set)

    # Must loop over each column to create an xarray DataArray for each
    for column_name in column_names:
        column_data = measurement_set.getcol(column_name)

        # UVW column must be split into u, v, and w
        if column_name == "UVW":
            subcolumns = [
                column_data[:, 0],
                column_data[:, 1],
                column_data[:, 2],
            ]
            subcolumn_names = ["U", "V", "W"]

            for data, name in zip(subcolumns, subcolumn_names):
                reshaped_column = reshape_column(
                    data,
                    data_shape,
                    time_indices,
                    baseline_indices,
                )

                # Create a DataArray instead of appending immediately to
                # Dataset so time coordinates can be updated
                xda = xr.DataArray(
                    reshaped_column,
                    dims=COLUMN_DIMENSIONS.get(name),
                )

                # Add the DataArray to the Dataset
                xds[COLUMN_TO_DATA_VARIABLE_NAMES.get(name)] = xda

        else:
            reshaped_column = reshape_column(
                column_data,
                data_shape,
                time_indices,
                baseline_indices,
            )

            # Create a DataArray instead of appending immediately to Dataset
            # so time coordinates can be updated
            xda = xr.DataArray(
                reshaped_column,
                dims=COLUMN_DIMENSIONS.get(column_name),
            )

            # Add the DataArray to the Dataset
            xds[COLUMN_TO_DATA_VARIABLE_NAMES.get(column_name)] = xda

    # Add column metadata at the end
    # Adding metadata to a variable means the variable must already exist
    for column_name in column_names:
        if column_name == "UVW":
            subcolumn_names = ["U", "V", "W"]

            for subcolumn_name in subcolumn_names:
                xds[
                    COLUMN_TO_DATA_VARIABLE_NAMES[subcolumn_name]
                ].attrs.update(
                    create_attribute_metadata(column_name, measurement_set)
                )

        else:
            xds[COLUMN_TO_DATA_VARIABLE_NAMES[column_name]].attrs.update(
                create_attribute_metadata(column_name, measurement_set)
            )

    ms_size_mb = get_dir_size(path=infile) / (1024 * 1024)

    # Ceiling division so that chunks are at least 100MB
    # Integer casting always returns a smaller number so chunks >100MB
    n_chunks = ms_size_mb // 100

    n_times = len(unique_times)

    # xr's chunk method requires rows_per_chunk as input not n_chunks
    times_per_chunk = 2 * n_times // n_chunks

    # Chunks method is number of pieces in the chunk
    # not the number of chunks. -1 gives a single chunk
    xds = xds.chunk(
        {
            "time": times_per_chunk,
            "frequency": -1,
            "baseline_id": -1,
            "polarization": -1,
        }
    )

    return xds.to_zarr(store=outfile, mode="w", compute=True)


def concatenate_stores(outfile_tmp, outfiles, times_per_chunk, client):
    """Concatenate multiple Zarr stores into a single Zarr store."""
    xds = xr.open_mfdataset(paths=outfiles, engine="zarr", parallel=True)
    xds = xds.chunk(
        {
            "time": times_per_chunk,
            "frequency": -1,
            "baseline_id": -1,
            "polarization": -1,
        }
    )

    synchronizer = zarr.ThreadSynchronizer()

    parallel_writes = xds.to_zarr(
        outfile_tmp,
        mode="w",
        compute=False,
        synchronizer=synchronizer,
        safe_chunks=False,
    )

    future = client.compute(parallel_writes)
    wait(future)

    for outfile in outfiles:
        shutil.rmtree(f"{outfile}")


def convert(infile, outfile, fits_in_memory=False, mem_avail=2.0, num_procs=4):
    """Converts a MeasurementSet to a Zarr format."""
    # Ensure JIT compilation before multiprocessing pool is spawned
    searchsorted_nb(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    isin_nb(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    antennas_to_indices(np.array([0, 1]), np.array([1, 2]))

    # Create temporary zarr store
    # Eventually clean-up by rechunking the zarr datastore
    outfile_tmp = outfile + ".tmp"

    measurement_set = tables.table(
        infile, readonly=True, lockoptions="autonoread"
    )

    # Get the unique timestamps
    # TODO pass time values to MS_to_zarr functions to reduce memory footprint
    # TODO use .col() instead of .getcol() for lazy loading and selecting
    time_values = measurement_set.getcol("TIME")
    unique_time_values = np.sort(pd.unique(time_values))

    # Get unique baseline indices
    baseline_ant1_id, baseline_ant2_id = get_invertible_indices(
        measurement_set
    )
    num_unique_baselines = len(baseline_ant1_id)

    # Base Dataset
    xds_base = xr.Dataset()

    # Add dimensions, coordinates and attributes here to prevent
    # repetition. Deep copy made for each loop iteration
    xds_base = create_coordinates(
        xds_base,
        measurement_set,
        unique_time_values,
        baseline_ant1_id,
        baseline_ant2_id,
        fits_in_memory,
    )

    column_names = measurement_set.colnames()
    columns_to_convert = []
    for col_name in column_names:
        if col_name in DATA_VARIABLE_COLUMNS:
            try:
                # Check that the column is populated
                if measurement_set.col(col_name)[0] is not None:
                    columns_to_convert.append(col_name)
            except RuntimeError:
                pass

    if fits_in_memory:
        ms_to_zarr_in_memory(
            xds_base,
            measurement_set,
            unique_time_values,
            num_unique_baselines,
            columns_to_convert,
            infile,
            outfile,
        )
    else:
        # Do not delete measurement_set from memory, otherwise must reload from
        # disk rather than copied in memory

        # Halve the total available memory for safety
        # Assumes the numpy arrays take up the same space in memory
        # as the measurement_set
        mem_avail_per_process = mem_avail * 0.5 * 1024.0**3 / num_procs
        ms_size = get_dir_size(infile)

        num_chunks = np.max([int(ms_size // mem_avail_per_process), num_procs])

        time_chunks = np.array_split(unique_time_values, num_chunks)

        # Ceiling division so that chunks are at least 100MB
        # Integer casting always returns a smaller number so chunks >100MB
        n_chunks = (ms_size / (1024 * 1024)) // 100

        # xr's chunk method requires rows_per_chunk as input not n_chunks
        times_per_chunk = len(unique_time_values) // n_chunks

        outfiles = []
        xds_list = []
        row_indices_list = []
        infiles = []
        antenna_length_list = []
        times_per_chunk_list = []
        data_variables_list = []
        column_dimension_list = []
        column_to_variable_list = []

        for i, time_chunk in enumerate(time_chunks):
            outfiles.append(outfile_tmp + str(i))
            xds_list.append(xds_base.copy(deep=True))
            row_indices_list.append(
                np.where(isin_nb(time_values, time_chunk))[0]
            )
            infiles.append(deepcopy(infile))
            antenna_length_list.append(deepcopy(num_unique_baselines))
            times_per_chunk_list.append(deepcopy(times_per_chunk))
            data_variables_list.append(deepcopy(columns_to_convert))
            column_dimension_list.append(deepcopy(COLUMN_DIMENSIONS))
            column_to_variable_list.append(
                deepcopy(COLUMN_TO_DATA_VARIABLE_NAMES)
            )

        del time_values
        # Must delete measurement_set here (especially for Unix systems)
        # Prevents multiprocessing from forking measurement_set to child
        # processes
        del measurement_set

        with mp.Pool(processes=num_procs, maxtasksperchild=1) as pool:
            pool.starmap(
                ms_chunk_to_zarr,
                zip(
                    xds_list,
                    infiles,
                    row_indices_list,
                    time_chunks,
                    antenna_length_list,
                    outfiles,
                    times_per_chunk_list,
                    data_variables_list,
                    column_dimension_list,
                    column_to_variable_list,
                ),
                chunksize=1,
            )

        ms_size_mb = get_dir_size(path=infile) / (1024 * 1024)

        # Ceiling division so that chunks are at least 100MB
        # Integer casting always returns a smaller number so chunks >100MB
        n_chunks = ms_size_mb // 100

        n_times = len(unique_time_values)

        # xr's chunk method requires rows_per_chunk as input not n_chunks
        times_per_chunk = 2 * n_times // n_chunks

        with LocalCluster(
            n_workers=1,
            processes=False,
            threads_per_worker=num_procs,
            memory_limit=f"{mem_avail}GiB",
        ) as cluster, Client(cluster) as client:
            concatenate_stores(outfile, outfiles, times_per_chunk, client)
