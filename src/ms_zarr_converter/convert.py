from casacore import tables
import numpy as np
import xarray as xr
import shutil
import os
from dask.distributed import LocalCluster, Client, wait
import multiprocessing as mp
import pandas as pd
import numba as nb
import zarr
from copy import deepcopy

# Use np.sort(pd.unique(arr)) not np.unique(arr) for large arrays: https://stackoverflow.com/questions/13688784/python-speedup-np-unique

@nb.njit(parallel=False, fastmath=True)
def searchsorted_nb(a, b):
    res = np.empty(len(b), np.intp)
    for i in nb.prange(len(b)):
        res[i] = np.searchsorted(a, b[i])
    return res

@nb.njit(parallel=False, fastmath=True)
def isin_nb(a: np.ndarray, b: np.ndarray):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in range(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)

@nb.njit(parallel=False, fastmath=True)
def antennas_to_indices(antenna1, antenna2):
    all_baselines = np.empty(antenna1.size, dtype=np.int32)
    
    for i in nb.prange(len(antenna1)):
        # This is a Cantor pairing funciton
        # max_num_antenna_pairs required to give expected ordering (20000 may be too small for certain SKA-low observations)
        # see https://math.stackexchange.com/questions/3212587/does-there-exist-a-pairing-function-which-preserves-ordering
        # and https://math.stackexchange.com/questions/3969617/what-pairing-function-coincides-with-the-g%C3%B6del-pairing-on-the-natural-numbers
        max_num_antenna_pairs = 20000
        all_baselines[i] = ((antenna1[i] + antenna2[i]) * (antenna1[i] + antenna2[i] + 1)) // 2 + max_num_antenna_pairs*antenna1[i]

    return all_baselines

@nb.njit(parallel=False, fastmath=True)
def invertible_indices(antenna1, antenna2):
    all_baselines = np.empty(antenna1.size, dtype=np.int32)
    
    for i in nb.prange(len(antenna1)):
        # This is a Cantor pairing funciton
        # max_num_antenna_pairs required to give expected ordering (20000 may be too small for certain SKA-low observations)
        # see https://math.stackexchange.com/questions/3212587/does-there-exist-a-pairing-function-which-preserves-ordering
        # and https://math.stackexchange.com/questions/3969617/what-pairing-function-coincides-with-the-g%C3%B6del-pairing-on-the-natural-numbers
        max_num_antenna_pairs = 20000
        all_baselines[i] = ((antenna1[i] + antenna2[i]) * (antenna1[i] + antenna2[i] + 1)) // 2 + antenna2[i]

    return all_baselines


@nb.njit(parallel=False, fastmath=True)
def indices_to_baseline_ids(unique_baselines):
    baseline_id1 = np.empty(unique_baselines.size, dtype=np.int32)
    baseline_id2 = np.empty(unique_baselines.size, dtype=np.int32)
    
    for i in nb.prange(len(unique_baselines)):
        # Inverse Cantor pairing funciton
        w = (np.sqrt(8*unique_baselines[i]+1) - 1) // 2
        t = (w * (w+1)) / 2
        y = unique_baselines[i] - t
        baseline_id2[i] = y
        baseline_id1[i] = w - y

    return (baseline_id1, baseline_id2)


# Named data variables (ie data)
# All other columns considered as metadata / coordinates
data_variable_columns = [
    "FLOAT_DATA",
    "DATA",
    "CORRECTED_DATA",
    "WEIGHT_SPECTRUM",
    "WEIGHT",
    "FLAG",
    "UVW",
    "TIME_CENTROID",
    "EXPOSURE",
    "FLAG_CATEGORY",
    "MODEL_DATA",
    "SIGMA",
    "ARRAY_ID",
    "DATA_DESC_ID",
    "FEED1",
    "FEED2",
    "FIELD_ID",
    "FLAG_ROW",
    "INTERVAL",
    "OBSERVATION_ID",
    "PROCESSOR_ID",
    "SCAN_NUMBER",
    "STATE_ID",
]


# Rename MS columns to a DataArray/ Dataset variable
column_to_data_variable_names = {
    "FLOAT_DATA": "SPECTRUM",
    "DATA": "VISIBILITY",
    "CORRECTED_DATA": "VISIBILITY_CORRECTED",
    "WEIGHT_SPECTRUM": "WEIGHT_SPECTRUM",
    "WEIGHT": "WEIGHT",
    "FLAG": "FLAG",
    "UVW": "UVW",
    "TIME_CENTROID": "TIME_CENTROID",
    "EXPOSURE": "EFFECTIVE_INTEGRATION_TIME",
    "U": "U",
    "V": "V",
    "W": "W",
    "FLAG_CATEGORY": "FLAG_CATEGORY",
    "MODEL_DATA": "MODEL_DATA",
    "SIGMA": "SIGMA",
    "ARRAY_ID": "ARRAY_ID",
    "DATA_DESC_ID": "DATA_DESC_ID",
    "FEED1": "FEED1",
    "FEED2": "FEED2",
    "FIELD_ID": "FIELD_ID",
    "FLAG_ROW": "FLAG_ROW",
    "INTERVAL": "INTERVAL",
    "OBSERVATION_ID": "OBSERVATION_ID",
    "PROCESSOR_ID": "PROCESSOR_ID",
    "SCAN_NUMBER": "SCAN_NUMBER",
    "STATE_ID": "STATE_ID",
}


# The dimensions of a given column
column_dimensions = {
    "DATA": ("time", "baseline_id", "frequency", "polarization"),
    "CORRECTED_DATA": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT_SPECTRUM": ("time", "baseline_id", "frequency", "polarization"),
    "WEIGHT": ("time", "baseline_id", "polarization"),
    "FLAG": ("time", "baseline_id", "frequency", "polarization"),
    "UVW": ("time", "baseline_id"),
    "U": ("time", "baseline_id"),
    "V": ("time", "baseline_id"),
    "W": ("time", "baseline_id"),
    "TIME_CENTROID": ("time", "baseline_id"),
    "EXPOSURE": ("time", "baseline_id"),
    "FLOAT_DATA": ("time", "baseline_id", "frequency", "polarization"),
    "FLAG_CATEGORY": ("time", "baseline_id", "frequency", "polarization"),
    "MODEL_DATA": ("time", "baseline_id", "frequency", "polarization"),
    "SIGMA": ("time", "baseline_id", "polarization"),
    "ARRAY_ID": ("time", "baseline_id"),
    "DATA_DESC_ID": ("time", "baseline_id"),
    "FEED1": ("time", "baseline_id"),
    "FEED2": ("time", "baseline_id"),
    "FIELD_ID": ("time", "baseline_id"),
    "FLAG_ROW": ("time", "baseline_id"),
    "INTERVAL": ("time", "baseline_id"),
    "OBSERVATION_ID": ("time", "baseline_id"),
    "PROCESSOR_ID": ("time", "baseline_id"),
    "SCAN_NUMBER": ("time", "baseline_id"),
    "STATE_ID": ("time", "baseline_id"),
}

# Rename coordinate columns to something sensible
column_to_coord_names = {
    "TIME": "time",
    "ANTENNA1": "baseline_antenna1_id",
    "ANTENNA2": "baseline_antenna2_id",
}


# Polarization IDs to Stokes parameter name
stokes_types = {
    0: "Undefined",
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY",
    13: "RX",
    14: "RY",
    15: "LX",
    16: "LY",
    17: "XR",
    18: "XL",
    19: "YR",
    20: "YL",
    21: "PP",
    22: "PQ",
    23: "QP",
    24: "QQ",
    25: "RCircular",
    26: "LCircular",
    27: "Linear",
    28: "Ptotal",
    29: "Plinear",
    30: "PFtotal",
    31: "PFlinear",
    32: "Pangle",
}



def get_dir_size(path="."):
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total



# Check that the time steps have a single unique value
def _check_interval_consistent(MeasurementSet):
    time_interval = pd.unique(MeasurementSet.getcol("INTERVAL"))
    assert len(time_interval) == 1, "Interval is not consistent."
    return time_interval[0]


# Check that the time steps have a single unique value
def _check_exposure_consistent(MeasurementSet):
    exposure_time = pd.unique(MeasurementSet.getcol("EXPOSURE"))
    assert len(exposure_time) == 1, "Exposure is not consistent."
    return exposure_time[0]


# Check that the FIELD_ID has a single unique value
def _check_single_field(MeasurementSet):
    field_id = pd.unique(MeasurementSet.getcol("FIELD_ID"))
    assert len(field_id) == 1, "More than one field present."
    return field_id[0]


# Create the DataArray coordinates and add various metadata
def create_coordinates(xds, MeasurementSet, unique_times, baseline_ant1_id, baseline_ant2_id, fits_in_memory):
    ###############################################################
    # Add metadata
    ###############################################################

    # Field subtable
    field = MeasurementSet.FIELD[0]

    # Assumes only a single FIELD_ID
    delay_direction = {
        "dims":"",
        "data": field.get("DELAY_DIR", [0]).tolist()[0],
        "attrs": {
            "units": "rad",
            "type": "sky_coord",
            "description": "Direction of delay center in right ascension and declination.",
        },
    }
    
    phase_direction = {
        "dims":"",
        "data": field.get("PHASE_DIR", [0]).tolist()[0],
        "attrs": {
            "units": "rad",
            "type": "sky_coord",
            "description": "Direction of phase center in right ascension and declination.",
        },
    }
    
    reference_direction = {
        "dims":"",
        "data": field.get("REFERENCE_DIR", [0]).tolist()[0],
        "attrs": {
            "units": "rad",
            "type": "sky_coord",
            "description": "Direction of reference direction in right ascension and declination. Used in single-dish to record the associated reference direction if position-switching has already been applied. For interferometric data, this is the original correlated field center, and may equal delay_direction or phase_direction.",
        },
    }
    
    name = field.get("NAME", "name")
    code = field.get("CODE", "1")
    
    if name == "":
        name = "name"
    if code == "":
        code = "1"
        
    field_id = _check_single_field(MeasurementSet)

    field_info = {
        "name": name,
        "code": code,
        "field_id": field_id,
    }
    
    xds.attrs["delay_direction"] = delay_direction
    xds.attrs["phase_direction"] = phase_direction
    xds.attrs["reference_direction"] = reference_direction
    xds.attrs["field_info"] = field_info
    
    

    ###############################################################
    # Add coordinates
    ###############################################################

    coords = {
        "baseline_antenna1_id": ("baseline_id", baseline_ant1_id),
        "baseline_antenna2_id": ("baseline_id", baseline_ant2_id),
        "baseline_id": np.arange(len(baseline_ant1_id)),
    }
    
    # If it doesn't fit in memory, populate time coordinate on-the-fly
    if fits_in_memory:
        coords["time"] = unique_times

    # Add frequency coordinates
    frequencies = MeasurementSet.SPECTRAL_WINDOW[0].get("CHAN_FREQ", [])

    # Remove NaN values
    coords["frequency"] = frequencies[~np.isnan(frequencies)]

    # Add polatization coordinates
    polarizations = MeasurementSet.POLARIZATION[0].get("CORR_TYPE", [])
    coords["polarization"] = np.vectorize(stokes_types.get)(polarizations)

    # Add named coordinates
    xds = xds.assign_coords(coords)

    ###############################################################
    # Add metadata to coordinates
    ###############################################################

    xds.frequency.attrs["reference_frequency"] = MeasurementSet.SPECTRAL_WINDOW[0].get(
        "REF_FREQUENCY", ""
    )
    xds.frequency.attrs["effective_channel_width"] = MeasurementSet.SPECTRAL_WINDOW[
        0
    ].get("EFFECTIVE_BW", "")[0]

    channel_widths = MeasurementSet.SPECTRAL_WINDOW[0].get("CHAN_WIDTH", "")

    if not isinstance(channel_widths, str):
        unique_chan_width = np.unique(
            channel_widths[np.logical_not(np.isnan(channel_widths))]
        )
        xds.frequency.attrs["channel_width"] = {
            "data": np.abs(unique_chan_width[0]),
            "attrs": {"type": "quanta", "units": "Hz"},
        }

    return xds


# Reshape a column
def reshape_column(
    column_data,
    cshape,
    time_indices,
    baselines,
):
    # full data is the maximum of the data shape and chunk shape dimensions for each time interval

    fulldata = np.full(cshape + column_data.shape[1:], np.nan, dtype=column_data.dtype)

    fulldata[time_indices, baselines] = column_data

    return fulldata


# All baseline pairs needed to reshape data columns
def get_baselines(MeasurementSet: tables.table) -> np.ndarray:
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = MeasurementSet.getcol("ANTENNA1", 0, -1), MeasurementSet.getcol(
        "ANTENNA2", 0, -1
    )
    baselines = np.array(
        [
            str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
            for ll in np.unique(np.hstack([ant1[:, None], ant2[:, None]]), axis=0)
        ]
    )

    return baselines



# Need to convert pairs of antennas into indices
# Integer indices required for array broadcasting
def get_baseline_pairs(MeasurementSet: tables.table) -> tuple:
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = MeasurementSet.getcol("ANTENNA1", 0, -1), MeasurementSet.getcol(
        "ANTENNA2", 0, -1
    )
    baselines = np.array(
        [
            str(ll[0]).zfill(3) + "_" + str(ll[1]).zfill(3)
            for ll in np.unique(np.hstack([ant1[:, None], ant2[:, None]]), axis=0)
        ]
    )

    baseline_ant1_id, baseline_ant2_id = np.array(
        [tuple(map(int, x.split("_"))) for x in baselines]
    ).T

    return (baseline_ant1_id, baseline_ant2_id)




def get_baseline_indices(MeasurementSet: tables.table) -> tuple:
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = MeasurementSet.getcol("ANTENNA1", 0, -1), MeasurementSet.getcol(
        "ANTENNA2", 0, -1
    )
    
    all_antenna_pairs = antennas_to_indices(ant1, ant2)
    
    # pd.unique is much faster than np.unique because it doesn't pre-sort
    # If len(unique_antenna_pairs) << len(all_antenna_pairs) then this is 
    # is a greater than 2x speedup
    unique_antenna_pairs = np.sort(pd.unique(all_antenna_pairs))

    # Compiled searchsort on pre-sorted arrays is ~2x faster than np.searchsorted
    baseline_indices = searchsorted_nb(unique_antenna_pairs, all_antenna_pairs)

    return baseline_indices

def get_invertible_indices(MeasurementSet: tables.table) -> tuple:
    # main table uses time x (antenna1,antenna2)
    ant1, ant2 = MeasurementSet.getcol("ANTENNA1", 0, -1), MeasurementSet.getcol(
        "ANTENNA2", 0, -1
    )
    
    unique_invertible_indices = np.sort(pd.unique(invertible_indices(ant1, ant2)))
    
    return indices_to_baseline_ids(unique_invertible_indices)
    


# Add column metadata for a single column
# Must be called after the DataArrays have been created
def create_attribute_metadata(column_name, MeasurementSet):
    
    attrs_metadata = {}

    if column_name in ["U", "V", "W"]:
        name = "UVW"
        column_description = MeasurementSet.getcoldesc(name)

        column_info = column_description.get("INFO", {})

        attrs_metadata["type"] = column_info.get("type", "None")
        attrs_metadata["ref"] = column_info.get("Ref", "None")

        attrs_metadata["units"] = column_description.get("keywords", {}).get(
            "QuantumUnits", ["None"]
        )[0]

    else:
        column_description = MeasurementSet.getcoldesc(column_name)

        column_info = column_description.get("INFO", {})

        attrs_metadata["type"] = column_info.get("type", "None")
        attrs_metadata["ref"] = column_info.get("Ref", "None")

        attrs_metadata["units"] = column_description.get("keywords", {}).get(
            "QuantumUnits", ["None"]
        )[0]

    return attrs_metadata


# Separate function to add time attributes. Must happen after the
# time coordinate is created
def add_time(xds, MeasurementSet):
    interval = _check_interval_consistent(MeasurementSet)
    exposure = _check_exposure_consistent(MeasurementSet)
    time_description = MeasurementSet.getcoldesc("TIME")

    xds.time.attrs["type"] = time_description.get("MEASINFO", {}).get("type", "None")

    xds.time.attrs["Ref"] = time_description.get("MEASINFO", {}).get("Ref", "None")

    xds.time.attrs["units"] = time_description.get("keywords", {}).get(
        "QuantumUnits", ["None"]
    )[0]

    xds.time.attrs["time_scale"] = (
        time_description.get("keywords", {}).get("MEASINFO", {}).get("Ref", "None")
    )

    xds.time.attrs["integration_time"] = interval

    xds.time.attrs["effective_integration_time"] = exposure
    return


def MS_chunk_to_zarr(
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
    
    # TODO refactor MS loading
    # Loading+querying uses a lot of memory
    with tables.table(infile, readonly=True, lockoptions="autonoread").selectrows(row_indices) as MeasurementSet_chunk:

        # Get dimensions of data
        time_indices = searchsorted_nb(times, MeasurementSet_chunk.getcol("TIME"))
        data_shape = (len(times), num_unique_baselines)

        baseline_indices = get_baseline_indices(MeasurementSet_chunk)

        # Must loop over each column to create an xarray DataArray for each
        for column_name in data_variable_columns:
            
            column_data = MeasurementSet_chunk.getcol(column_name)

            # UVW column must be split into u, v, and w
            if column_name == "UVW":
                subcolumns = [column_data[:, 0], column_data[:, 1], column_data[:, 2]]
                subcolumn_names = ["U", "V", "W"]

                for data, name in zip(subcolumns, subcolumn_names):
                    reshaped_column = reshape_column(
                        data,
                        data_shape,
                        time_indices,
                        baseline_indices,
                    )

                    # Create a DataArray instead of appending immediately to Dataset
                    # so time coordinates can be updated
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

                # Create a DataArray instead of appending immediately to Dataset
                # so time coordinates can be updated
                xda = xr.DataArray(
                    reshaped_column,
                    dims=column_dimensions.get(column_name),
                ).assign_coords(time=("time", times))

                # Add the DataArray to the Dataset
                xds[column_to_data_variable_names.get(column_name)] = xda

                # Add column metadata at the end
                # Adding metadata to a variable means the variable must already exist

                xds[column_to_data_variable_names[column_name]].attrs.update(
                    create_attribute_metadata(column_name, MeasurementSet_chunk)
                    )
            
        xds = xds.chunk({"time": times_per_chunk, "frequency":-1, "baseline_id":-1, "polarization":-1})
            
        add_time(xds, MeasurementSet_chunk)
        xds.to_zarr(store=outfile, mode="w", compute=True)
    return


def MS_to_zarr_in_memory(
    xds,
    MeasurementSet,
    unique_times,
    num_unique_baselines,
    column_names,
    infile,
    outfile,
):


    # Get dimensions of data
    time_indices = searchsorted_nb(unique_times, MeasurementSet.getcol("TIME"))
    data_shape = (len(unique_times), num_unique_baselines)

    baseline_indices = get_baseline_indices(MeasurementSet)

    # Must loop over each column to create an xarray DataArray for each
    for column_name in column_names:
        column_data = MeasurementSet.getcol(column_name)

        # UVW column must be split into u, v, and w
        if column_name == "UVW":
            subcolumns = [column_data[:, 0], column_data[:, 1], column_data[:, 2]]
            subcolumn_names = ["U", "V", "W"]

            for data, name in zip(subcolumns, subcolumn_names):
                reshaped_column = reshape_column(
                    data,
                    data_shape,
                    time_indices,
                    baseline_indices,
                )

                # Create a DataArray instead of appending immediately to Dataset
                # so time coordinates can be updated
                xda = xr.DataArray(
                    reshaped_column,
                    dims=column_dimensions.get(name),
                )

                # Add the DataArray to the Dataset
                xds[column_to_data_variable_names.get(name)] = xda

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
                dims=column_dimensions.get(column_name),
            )

            # Add the DataArray to the Dataset
            xds[column_to_data_variable_names.get(column_name)] = xda

    # Add column metadata at the end
    # Adding metadata to a variable means the variable must already exist
    for column_name in column_names:
        if column_name == "UVW":
            subcolumn_names = ["U", "V", "W"]

            for subcolumn_name in subcolumn_names:
                xds[column_to_data_variable_names[subcolumn_name]].attrs.update(
                    create_attribute_metadata(column_name, MeasurementSet)
                )

        else:
            xds[column_to_data_variable_names[column_name]].attrs.update(
                create_attribute_metadata(column_name, MeasurementSet)
            )
        
    MS_size_MB = get_dir_size(path=infile) / (1024*1024)
    
    # Ceiling division so that chunks are at least 100MB
    # Integer casting always returns a smaller number so chunks >100MB
    n_chunks = MS_size_MB // 100
    
    ntimes = len(unique_times)

    # xr's chunk method requires rows_per_chunk as input not n_chunks
    times_per_chunk = 2 * ntimes // n_chunks

    # Chunks method is number of pieces in the chunk
    # not the number of chunks. -1 gives a single chunk
    xds = xds.chunk({"time": times_per_chunk, "frequency":-1, "baseline_id":-1, "polarization":-1})
        
    return xds.to_zarr(store=outfile, mode="w", compute=True)


def concatenate_stores(infile, outfile_tmp, outfiles, times_per_chunk, client):
    
    xds = xr.open_mfdataset(paths=outfiles, engine="zarr", parallel=True)
    xds = xds.chunk({"time": times_per_chunk, "frequency":-1, "baseline_id":-1, "polarization":-1})
    
    synchronizer = zarr.ThreadSynchronizer()
    
    parallel_writes = xds.to_zarr(outfile_tmp, mode="w", compute=False, synchronizer=synchronizer, safe_chunks=False)
    
    future = client.compute(parallel_writes)
    wait(future)
    
    for outfile in outfiles:
        shutil.rmtree(f"{outfile}")
        
    return


def convert(infile, outfile, fits_in_memory=False, mem_avail=2., num_procs=4):
    # Ensure JIT compilation before multiprocessing pool is spawned
    searchsorted_nb(np.array([0.,1.]), np.array([0.,1.]))
    isin_nb(np.array([0.,1.]), np.array([0.,1.]))
    antennas_to_indices(np.array([0,1]), np.array([1,2]))
    
    
    # Create temporary zarr store
    # Eventually clean-up by rechunking the zarr datastore
    outfile_tmp = outfile + ".tmp"

        
    MeasurementSet = tables.table(infile, readonly=True, lockoptions="autonoread")
    
    # Initial checks
    time_interval = _check_interval_consistent(MeasurementSet)
    exposure_time = _check_exposure_consistent(MeasurementSet)
    field_id = _check_single_field(MeasurementSet)

    # Get the unique timestamps
    # TODO pass time values to MS_to_zarr functions to reduce memory footprint
    # TODO use .col() instead of .getcol() for lazy loading and selecting
    time_values = MeasurementSet.getcol("TIME")
    unique_time_values = np.sort(pd.unique(time_values))

    # Get unique baseline indices
    baseline_ant1_id, baseline_ant2_id = get_invertible_indices(MeasurementSet)
    num_unique_baselines = len(baseline_ant1_id)

    # Base Dataset
    xds_base = xr.Dataset()

    # Add dimensions, coordinates and attributes here to prevent
    # repetition. Deep copy made for each loop iteration
    xds_base = create_coordinates(
        xds_base, MeasurementSet, unique_time_values, baseline_ant1_id, baseline_ant2_id, fits_in_memory
    )

    column_names = MeasurementSet.colnames()
    
    columns_to_convert = []
    
    for col_name in column_names:
        if col_name in data_variable_columns:
            try:
                # Check that the column is populated
                if MeasurementSet.col(col_name)[0] is not None:
                    columns_to_convert.append(col_name)
                    
            except:
                pass

    if fits_in_memory:
        MS_to_zarr_in_memory(
            xds_base,
            MeasurementSet,
            unique_time_values,
            num_unique_baselines,
            columns_to_convert,
            infile,
            outfile,
        )
        
    else:
        # Do not delete MeasurementSet from memory, otherwise must reload from disk rather than copied in memory
        
        # Halve the total available memory for safety
        # Assumes the numpy arrays take up the same space in memory
        # as the MeasurementSet
        mem_avail_per_process = mem_avail * 0.5 * 1024.**3 / num_procs
        MS_size = get_dir_size(infile)
        
        num_chunks = np.max([int(MS_size // mem_avail_per_process), num_procs])
        
        time_chunks = np.array_split(unique_time_values, num_chunks)
    
        # Ceiling division so that chunks are at least 100MB
        # Integer casting always returns a smaller number so chunks >100MB
        n_chunks = (MS_size / (1024*1024)) // 100

        # xr's chunk method requires rows_per_chunk as input not n_chunks
        times_per_chunk = len(unique_time_values) // n_chunks

        
        outfiles = []
        xds_list = []
        row_indices_list = []
        infiles = []
        antenna_length_list = []
        column_name_list = []
        times_per_chunk_list = []
        data_variables_list = []
        column_dimension_list = []
        column_to_variable_list = []
        
        for i, time_chunk in enumerate(time_chunks):
            outfiles.append(outfile_tmp + str(i))
            xds_list.append(xds_base.copy(deep=True))
            row_indices_list.append(np.where(isin_nb(time_values, time_chunk))[0])
            infiles.append(deepcopy(infile))
            antenna_length_list.append(deepcopy(num_unique_baselines))
            times_per_chunk_list.append(deepcopy(times_per_chunk))
            data_variables_list.append(deepcopy(columns_to_convert))
            column_dimension_list.append(deepcopy(column_dimensions))
            column_to_variable_list.append(deepcopy(column_to_data_variable_names))
        

        del time_values
        # Must delete MeasurementSet here (especially for Unix systems)
        # Prevents multiprocessing from forking MeasurementSet to child processes
        del MeasurementSet
        
        
        with mp.Pool(processes=num_procs, maxtasksperchild=1) as pool:
            
            pool.starmap(MS_chunk_to_zarr, zip(xds_list, 
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
            
        MS_size_MB = get_dir_size(path=infile) / (1024*1024)
    
        # Ceiling division so that chunks are at least 100MB
        # Integer casting always returns a smaller number so chunks >100MB
        n_chunks = MS_size_MB // 100
        
        ntimes = len(unique_time_values)

        # xr's chunk method requires rows_per_chunk as input not n_chunks
        times_per_chunk = 2 * ntimes // n_chunks
        
        with LocalCluster(n_workers=1,
                          processes=False,
                          threads_per_worker=num_procs,
                          memory_limit=f"{mem_avail}GiB",
                        ) as cluster, Client(cluster) as client:
            
            concatenate_stores(infile, outfile, outfiles, times_per_chunk, client)

        return
    