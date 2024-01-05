# MeasurementSet to xarray

This is a prototype converter from MeasurementSet v2.0 to an xarray Dataset.

## Installation

The package can be loaded as a pip package. To install:

1) Clone the project `git clone https://github.com/sstansill/msconverter`
2) `cd msconverter`
3) Create an empty conda environment using `conda create -n msconverter python=3.11`
4) Activate the conda environment `conda activate msconverter`
5) Install pip dependencies `pip install -e .`
6) Inside a Python shell, use the converter with `from msconverter import convert`
7) `convert.convert(infile, outfile, compress=True)`

---

## Usage

Minimal Python script

```
from msconverter import convert

infile = "/path/to/MeasurementSet"
outfile = "/path/to/ZarrStore"

if __name__ == "__main__":
    convert.convert(infile, outfile, compress=True, fits_in_memory=False, mem_avail=4., num_procs=4)

```

---

## Options

- `compress` option can be used to toggle whether data is compressed using the lossless Zstandard compression algorithm.
- `fits_in_memory` should be set to `False` if the size of the MeasurementSet to be converted is more than half of the available system memory (both the MeasurementSet and the xarray DataSet will be held in memory)
- `num_procs` is the number of processes to be spawned by the multiprocessing library
- `mem_avail` is the total memory to be used by the Python script (unstable)

Note: the `mem_avail` option is not a hard limit. `mem_avail` will be quickly exceeded for very large MeasurementSets (the time column is held by each worker process)

---

## Performance

- Machine: CSD3 (cascade lake node)
- `num_procs=4`
- `mem_avail=6.`
- `compress=True `
- `#SBATCH --cpus-per-task=4`
- `#SBATCH --mem=32000`
- Dataset: `"/rds/projects/rds-sdhp-S7lLL7eOZIg/LOFAR-test-data-set/L628614_SAP004_SB340_uv_001.MS"`
- Dataset size: 71G
- Zarr store size on disk: 22G
- Conversion time: 11 mins 34.42s
