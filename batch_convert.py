from typing import List, Dict
import re
import glob
import os
from pathlib import Path
from dataclasses import dataclass
import logging
from multiprocessing import Pool
import time
import logging

from smry2parquet import smry2parquet

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


#paths = ["../webviz-subsurface-testdata/reek_history_match/realization-*/iter-0/eclipse/model/*.UNSMRY"]
paths = ["../../webviz_testdata/reek_history_match_large/realization-*/iter-0/eclipse/model/*.UNSMRY"]

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

realidxregexp = re.compile(r"realization-(\d+)")

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

# Limit number of files to process
#files_to_process = files_to_process[1:10]


def process_one_file(entry: FileEntry):
    LOGGER.info(f"real={entry.real}: {entry.filename}")
    output_filename = str(output_dir / f"summary_r{entry.real:03}.parquet")
    smry2parquet(entry.filename, output_filename, write_extra_feather=True)



LOGGER.info("Doing BATCH conversion SMRY -> Parquet")
start_s = time.perf_counter()

for entry in files_to_process:
    process_one_file(entry)

"""
# Experiment with multiple processes
# Significant speedups to be had here for multiple cores!!
with Pool() as pool:
    pool.map(process_one_file, files_to_process)
"""

LOGGER.info(f"Conversion finished in {(time.perf_counter() - start_s):.2f}s")
