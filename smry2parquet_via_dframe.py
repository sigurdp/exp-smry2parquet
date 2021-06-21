from typing import List, Dict
import argparse
from pathlib import Path
import logging
import logging.config
import json
import time

import pandas as pd
import pyarrow as pa
import pyarrow.feather
import pyarrow.parquet

from ecl.summary import EclSum



# -------------------------------------------------------------------------
def _set_date_column_type_to_timestamp_ms(schema: pa.Schema) -> pa.Schema:
    """ Returns new schema with the data type for the DATE column set to timestamp[ms]
    """
    dt_timestamp_ms = pa.timestamp("ms")
    indexof_date_field = schema.get_field_index("DATE")

    types = schema.types
    types[indexof_date_field] = dt_timestamp_ms

    field_list = zip(schema.names, types)
    return pa.schema(field_list)


# -------------------------------------------------------------------------
def _create_float_downcasting_schema(schema: pa.Schema) -> pa.Schema:
    """ Returns schema with all 64bit floats downcasted to 32bit float
    """
    dt_float64 = pa.float64()
    dt_float32 = pa.float32()
    types = schema.types
    for idx, typ in enumerate(types):
        if typ == dt_float64:
            types[idx] = dt_float32

    field_list = zip(schema.names, types)
    return pa.schema(field_list)


# -------------------------------------------------------------------------
def _set_metadata_per_field(schema: pa.Schema, smry_meta_dict: Dict[str, dict]) -> pa.Schema:
    # Strangely there seems to be no way of directly getting the number of fields in
    # the schema, nor a list of the fields
    new_field_list: List[pa.Field] = []
    field_count = len(schema.names)
    for idx in range(0, field_count):
        field = schema.field(idx)
        if field.name in smry_meta_dict:
            field = field.with_metadata({b"smry_meta": json.dumps(smry_meta_dict[field.name])})

        new_field_list.append(field)

    return pa.schema(new_field_list)


# -------------------------------------------------------------------------
def _create_smry_meta_dict(eclsum: EclSum, column_names: List[str]) -> Dict[str, dict]:
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

    eclsum = EclSum(smry_filename, lazy_load=False)

    # EclSum.pandas_frame() crashes if the SMRY data has timestamps beyond 2262
    # See: https://github.com/equinor/ecl/issues/802
    df: pd.DataFrame = eclsum.pandas_frame()

    # This could be a possible work-around to the crash above, but both numerical and 
    # performance impacts must be investigated.
    #ecldates = eclsum.dates
    #df: pd.DataFrame = eclsum.pandas_frame(time_index=ecldates)

    logger.debug("DataFrame shape: %s", df.shape)
    logger.debug("DataFrame head():\n%s", df.head())
    logger.debug("DataFrame tail():\n%s", df.tail())

    df.index.rename("DATE", inplace=True)
    df.reset_index(inplace=True)

    schema: pa.Schema = pa.Schema.from_pandas(df, preserve_index=False)
    schema = _set_date_column_type_to_timestamp_ms(schema)

    # Hopefully ecl will be able to deliver data with float precision directly in the 
    # future, ref issue https://github.com/equinor/ecl/issues/797
    # In the meantime we try and downcast the double columns to float32 
    schema = _create_float_downcasting_schema(schema)

    column_names = schema.names
    column_names.remove("DATE")
    smry_meta_dict = _create_smry_meta_dict(eclsum, column_names)

    schema = _set_metadata_per_field(schema, smry_meta_dict)

    table: pa.Table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    # Wipe out any Pandas related metadata from the schema
    #table = table.replace_schema_metadata(None)

    # Write the smry metadata to the schema.
    # Will completely replace any existing metadata in the table.
    new_schema_metadata = { b"smry_meta": json.dumps(smry_meta_dict)}
    table = table.replace_schema_metadata(new_schema_metadata)

    # print("table.schema.names:", table.schema.names)
    # print(table.schema.field("DATE"))
    # print(table.schema.metadata)
    # print("table.schema.pandas_metadata", table.schema.pandas_metadata)
    # print(table.schema)

    return table


# -------------------------------------------------------------------------
def smry2parquet(smry_filename: str, parquet_filename: str, write_extra_feather: bool) -> pa.Table:
    lap_s = time.perf_counter()
    logger.debug(f"Reading input SMRY data from: {smry_filename}")
    table: pa.Table = _load_smry_into_table(smry_filename)
    logger.debug(f"Reading input took {(time.perf_counter() - lap_s):.2f}s")

    lap_s = time.perf_counter()
    logger.debug(f"Writing parquet file to: {parquet_filename}")
    pa.parquet.write_table(table, parquet_filename)
    pa.parquet.write_table(table, parquet_filename, version="2.0", compression="ZSTD")
    logger.debug(f"Parquet write took {(time.perf_counter() - lap_s):.2f}s")

    # For testing/comparison purposes, we can also write to feather/arrow
    if write_extra_feather:
        lap_s = time.perf_counter()
        arrow_filename = Path(parquet_filename).with_suffix(".arrow")
        logger.debug(f"Writing arrow/feather file to: {arrow_filename}")
        pa.feather.write_feather(table, dest=arrow_filename)
        #pa.feather.write_feather(table, dest=arrow_filename, compression="zstd")
        logger.debug(f"Arrow/feather write took {(time.perf_counter() - lap_s):.2f}s")


# -------------------------------------------------------------------------
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("smry_file", help="Input UNSMRY file to convert")
    parser.add_argument("output", help="Output Parquet file name")

    args = parser.parse_args()

    smry_filename = args.smry_file
    parquet_filename = args.output

    #smry_filename = "../webviz-subsurface-testdata/reek_history_match/realization-0/iter-0/eclipse/model/5_R001_REEK-0.UNSMRY"
    #smry_filename = "../../webviz_testdata/reek_history_match_large/realization-2/iter-0/eclipse/model/R001_REEK-2.UNSMRY"
    #smry_filename = "./testdata/DROGON-0.UNSMRY"

    start_s = time.perf_counter()
    logger.info(f"Converting SMRY to Parquet (VIA DATAFRAME): smry={smry_filename}  parquet={parquet_filename}")

    smry2parquet(smry_filename, parquet_filename, write_extra_feather=True)

    logger.info(f"Conversion finished in {(time.perf_counter() - start_s):.2f}s")


