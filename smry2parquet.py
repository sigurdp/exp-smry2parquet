from typing import List, Dict, Iterable, Set
import argparse
from pathlib import Path
import logging
import json
import time

import pyarrow as pa
import pyarrow.feather
import pyarrow.parquet

from ecl.summary import EclSum, EclSumKeyWordVector

LOGGER = logging.getLogger(__name__)


# -------------------------------------------------------------------------
def _create_smry_meta_dict(eclsum: EclSum, column_names: Iterable[str]) -> Dict[str, dict]:
    """ Builds dictionary containing metadata for all the specified columns
    """
    smry_meta = {}

    for col_name in column_names:
        col_meta = {}
        col_meta["unit"] = eclsum.unit(col_name)
        col_meta["is_total"] = eclsum.is_total(col_name)
        col_meta["is_rate"] = eclsum.is_rate(col_name)
        col_meta["is_historical"] = eclsum.smspec_node(col_name).is_historical()
        col_meta["keyword"] = eclsum.smspec_node(col_name).keyword
        col_meta["wgname"] = eclsum.smspec_node(col_name).wgname

        num = eclsum.smspec_node(col_name).get_num()
        if num is not None:
            col_meta["get_num"] = num

        smry_meta[col_name] = col_meta

    return smry_meta


# -------------------------------------------------------------------------
def _load_smry_into_table(smry_filename: str) -> pa.Table:
    """
    Reads data from SMRY file into PyArrow Table.
    DATE column is stored as an Arrow timetamp with ms resolution, timestamp[ms]
    All numeric columns will be stored as 32 bit float
    Summary meta data will be attached per field/column of the table's schema under the
    'smry_meta' key
    """

    eclsum = EclSum(smry_filename, lazy_load=False)

    # Unclear what the difference between these two is, but it seems that
    # EclSum.pandas_frame() internally uses EclSumKeyWordVector
    # For now, we go via a set to prune out duplicate entries being returned by EclSumKeyWordVector,
    # see: https://github.com/equinor/ecl/issues/816#issuecomment-865881283
    column_names: Set[str] = set(EclSumKeyWordVector(eclsum, add_keywords = True))
    #column_names = eclsum.keys()

    # Fetch the dates as a numpy array with ms resolution
    np_dates_ms = eclsum.numpy_dates

    smry_meta_dict = _create_smry_meta_dict(eclsum, column_names)

    # Datatypes to use for DATE column and all the numeric columns
    dt_timestamp_ms = pa.timestamp("ms")
    dt_float32 = pa.float32()

    # Build schema for the table
    field_list: List[pa.Field] = []
    field_list.append(pa.field("DATE", dt_timestamp_ms))
    for colname in column_names:
        field_metadata = { b"smry_meta": json.dumps(smry_meta_dict[colname])}
        field_list.append(pa.field(colname, dt_float32, metadata=field_metadata))

    schema = pa.schema(field_list)

    # Now extract all the summary vectors one by one
    # We do this through EclSum.numpy_vector() instead of EclSum.pandas_frame() since
    # the latter throws an exception if the SMRY data has timestamps beyond 2262,
    # see: https://github.com/equinor/ecl/issues/802
    column_arrays = [ np_dates_ms ]

    for colname in column_names:
        colvector = eclsum.numpy_vector(colname)
        column_arrays.append(colvector)

    table = pa.table(column_arrays, schema=schema)

    return table


# -------------------------------------------------------------------------
def smry2parquet(smry_filename: str, parquet_filename: str, write_extra_feather: bool) -> pa.Table:
    lap_s = time.perf_counter()
    LOGGER.debug(f"Reading input SMRY data from: {smry_filename}")
    table: pa.Table = _load_smry_into_table(smry_filename)
    LOGGER.debug(f"Reading input took {(time.perf_counter() - lap_s):.2f}s")

    lap_s = time.perf_counter()
    LOGGER.debug(f"Writing parquet file to: {parquet_filename}")
    pa.parquet.write_table(table, parquet_filename)
    #pa.parquet.write_table(table, parquet_filename, version="2.0", compression="ZSTD")
    LOGGER.debug(f"Parquet write took {(time.perf_counter() - lap_s):.2f}s")

    # For testing/comparison purposes, we can also write to feather/arrow
    if write_extra_feather:
        lap_s = time.perf_counter()
        arrow_filename = Path(parquet_filename).with_suffix(".arrow")
        LOGGER.debug(f"Writing arrow/feather file to: {arrow_filename}")
        pa.feather.write_feather(table, dest=arrow_filename)
        #pa.feather.write_feather(table, dest=arrow_filename, compression="zstd")
        LOGGER.debug(f"Arrow/feather write took {(time.perf_counter() - lap_s):.2f}s")


# -------------------------------------------------------------------------
if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("smry_file", help="Input UNSMRY file to convert")
    parser.add_argument("output", help="Output Parquet file name")

    args = parser.parse_args()

    smry_filename = args.smry_file
    parquet_filename = args.output

    start_s = time.perf_counter()
    LOGGER.info(f"Converting SMRY to Parquet: smry={smry_filename}  parquet={parquet_filename}")

    smry2parquet(smry_filename, parquet_filename, write_extra_feather=True)

    LOGGER.info(f"Conversion finished in {(time.perf_counter() - start_s):.2f}s")
