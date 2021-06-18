from pathlib import Path
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet
import pyarrow.feather


INPUT_FILENAME = "output/summary.parquet"

if Path(INPUT_FILENAME).suffix == ".parquet":
    print("===================================")
    print("QUERIES AGAINST PARQUET FILE")
    print("===================================")

    # See: https://arrow.apache.org/docs/python/parquet.html
    parquet_file = pyarrow.parquet.ParquetFile(INPUT_FILENAME)
    print(parquet_file.metadata.row_group(0).column(0).statistics)
    print(parquet_file.metadata.row_group(0).column(1).statistics)
    print(parquet_file.metadata)


if Path(INPUT_FILENAME).suffix == ".parquet":
    table = pa.parquet.read_table(INPUT_FILENAME)
else:
    table = pa.feather.read_table("INPUT_FILENAME")


# Just select a few columns for debugging
table = table.select(["DATE", "FOPR", "TCPU"])


print("===================================")
print("QUERIES AGAINST ARROW TABLE")
print("===================================")

print("\ntable.schema")
print(table.schema)

#print("\ntable.schema.metadata")
#print(table.schema.metadata)

print("\ntable.shape", table.shape)

arrow_date = table["DATE"][0]
print("\narrow_date:", type(arrow_date), arrow_date.type, arrow_date)


schema_smry_meta = json.loads(table.schema.metadata[b"smry_meta"])
col_name = "FOPR"
print(f"schema metadata for {col_name}: {schema_smry_meta[col_name]}")
print(f" field metadata for {col_name}: {json.loads(table.field(col_name).metadata[b'smry_meta'])}")

col_name = table.schema.names[-1]
print(f"schema metadata for {col_name}: {schema_smry_meta[col_name]}")
print(f" field metadata for {col_name}: {json.loads(table.field(col_name).metadata[b'smry_meta'])}")


print("\n")
print("===================================")
print("QUERIES AGAINST PANDAS DATAFRAME")
print("===================================")

df = table.to_pandas(timestamp_as_object=True)
print(df.head())

pandasdate = df["DATE"][0]
print("\npandasdate", type(pandasdate), pandasdate)
