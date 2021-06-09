import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet
import pyarrow.feather


table = pa.parquet.read_table("output/summary.parquet")
#table = pa.feather.read_table("output/summary.arrow")

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

smry_meta = json.loads(table.schema.metadata[b"smry_meta"])
col_name = "FOPR"
print(f"metadata for {col_name}: {smry_meta[col_name]}")
col_name = table.schema.names[-1]
print(f"metadata for {col_name}: {smry_meta[col_name]}")


print("\n")
print("===================================")
print("QUERIES AGAINST PANDAS DATAFRAME")
print("===================================")

df = table.to_pandas(timestamp_as_object=True)
print(df.head())

pandasdate = df["DATE"][0]
print("\npandasdate", type(pandasdate), pandasdate)
