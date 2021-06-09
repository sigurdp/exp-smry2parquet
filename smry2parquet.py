from typing import List, Dict
from pathlib import Path
import logging
import logging.config
import json

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
def load_smry_into_table(smry_filename: str) -> pa.Table:

    eclsum = EclSum(smry_filename, lazy_load=False)

    # EclSum.pandas_frame() crashes if the SMRY data has timestamps beyond 2262
    df: pd.DataFrame = eclsum.pandas_frame()

    # This could be a possible work-around to the crash above, but both numerical and 
    # performance impacts must be investigated.
    #ecldates = eclsum.dates
    #df: pd.DataFrame = eclsum.pandas_frame(time_index=ecldates)

    print(df.shape)
    print(df.head())
    print(df.tail())

    df.index.rename("DATE", inplace=True)
    df.reset_index(inplace=True)

    schema: pa.Schema = pa.Schema.from_pandas(df, preserve_index=False)
    schema = _set_date_column_type_to_timestamp_ms(schema)

    # Hopefully ecl will be able to deliver data with float precision directly in the 
    # future, ref issue https://github.com/equinor/ecl/issues/797
    # In the meantime we try and downcast the double columns to float32 
    schema = _create_float_downcasting_schema(schema)

    table: pa.Table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

    column_names = schema.names
    column_names.remove("DATE")
    smry_meta_dict = _create_smry_meta_dict(eclsum, column_names)

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
if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    print("Starting...")

    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)

    #smry_filename = "../webviz-subsurface-testdata/reek_history_match/realization-0/iter-0/eclipse/model/5_R001_REEK-0.UNSMRY"
    smry_filename = "../../webviz_testdata/reek_history_match_large/realization-2/iter-0/eclipse/model/R001_REEK-2.UNSMRY"
    #smry_filename = "./testdata/DROGON-0.UNSMRY"

    parquet_filename = str(output_dir / "summary.parquet")
    arrow_filename = str(output_dir / "summary.arrow")

    table: pa.Table = load_smry_into_table(smry_filename)

    pa.parquet.write_table(table, parquet_filename)
    #pa.parquet.write_table(table, parquet_filename, version="2.0", compression="ZSTD")

    pa.feather.write_feather(table, dest=arrow_filename)
    #pa.feather.write_feather(table, dest=arrow_filename, compression="zstd")


