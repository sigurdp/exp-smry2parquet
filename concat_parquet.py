from typing import List, Dict
import re
import glob
import os
from pathlib import Path
from dataclasses import dataclass
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

import pyarrow as pa
import pyarrow.feather
import pyarrow.parquet
import numpy as np


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


paths = ["./output/summary_r*.parquet"]
#paths = ["./output/summary_r*.arrow"]

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

realidxregexp = re.compile(r"summary_r(\d+)")

globbedpaths = [glob.glob(path) for path in paths]
globbedpaths = sorted(list({item for sublist in globbedpaths for item in sublist}))


@dataclass
class FileEntry:
    real: int;
    filename: str;

files_to_process: List[FileEntry] = []

for path in globbedpaths:
    real = None
    for path_comp in reversed(path.split(os.path.sep)):
        realmatch = re.match(realidxregexp, path_comp)
        if realmatch:
            real = int(realmatch.group(1))
            files_to_process.append(FileEntry(real=real, filename=path))
            break

files_to_process = sorted(files_to_process, key=lambda e: e.real)

LOGGER.info("Doing CONCATENATION")
start_s = time.perf_counter()


def read_and_build_table_for_one_real(entry: FileEntry) -> pa.Table:
    LOGGER.info(f"real={entry.real}: {entry.filename}")

    if Path(entry.filename).suffix == ".parquet":
        table = pa.parquet.read_table(entry.filename)
    else:
        table = pa.feather.read_table(entry.filename)

    #table = table.select(["DATE", "FOPR", "TCPU", "BWIP:28,20,13"])

    table = table.add_column(1, "REAL", pa.array(np.full(table.num_rows, entry.real)))
    LOGGER.info(f"table shape: {table.shape}")

    return table


lap_s = time.perf_counter()

table_list = []


for entry in files_to_process:
    table = read_and_build_table_for_one_real(entry)
    table_list.append(table)


"""
#with ProcessPoolExecutor() as executor:
with ThreadPoolExecutor() as executor:
    futures = executor.map(read_and_build_table_for_one_real, files_to_process)
for f in futures:
    table_list.append(f)
"""

LOGGER.info(f"All input files read into memory in {(time.perf_counter() - lap_s):.2f}s")


unique_column_names = set()
for table in table_list:
    unique_column_names.update(table.schema.names)
LOGGER.info(f"number of unique column names: {len(unique_column_names)}")


LOGGER.info(f"number of tables to concatenate: {len(table_list)}")

lap_s = time.perf_counter()

# Need to investigate this further
# The default promote=False requires all schemas to be the same
# What should really happen if the realizations have different columns?
#combined_table = pa.concat_tables(table_list)
combined_table = pa.concat_tables(table_list, promote=True)
LOGGER.info(f"combined table shape: {combined_table.shape}")
LOGGER.info(f"In-memory concatenation took {(time.perf_counter() - lap_s):.2f}s")


lap_s = time.perf_counter()
output_parquet_filename = str(output_dir / "concat.parquet")
LOGGER.info(f"Writing parquet output to: {output_parquet_filename}")
pa.parquet.write_table(combined_table, output_parquet_filename)
LOGGER.info(f"Writing parquet took {(time.perf_counter() - lap_s):.2f}s")

lap_s = time.perf_counter()
output_feather_filename = str(output_dir / "concat.arrow")
LOGGER.info(f"Writing feather output to: {output_feather_filename}")
pa.feather.write_feather(combined_table, dest=output_feather_filename)
LOGGER.info(f"Writing feather took {(time.perf_counter() - lap_s):.2f}s")

LOGGER.info(f"DONE! total time was {(time.perf_counter() - start_s):.2f}s")

#df = combined_table.to_pandas(timestamp_as_object=True)
#print(df.head())

