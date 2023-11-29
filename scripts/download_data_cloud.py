# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download GNoME data to a chosen directory."""

from collections.abc import Sequence
import os
from absl import app
from absl import flags
from google.cloud import storage

_DATA_DIR = flags.DEFINE_string(
    name="data_dir",
    default="data",
    help="Location to copy downloaded data.",
)

BUCKET_NAME = "gdm_materials_discovery"
FOLDER_NAME = "gnome_data"
FILES = (
    "stable_materials_hull.csv",
    "stable_materials_r2scan.csv",
    "stable_materials_summary.csv",
    "by_composition.zip",
    "by_id.zip",
    "by_reduced_formula.zip",
)


def copy_blob(bucket, blob_name, copy_dir):
  """Copy a file from Google Cloud storage."""
  blob = bucket.get_blob(blob_name)
  output_filename = os.path.join(copy_dir, blob_name)
  blob.download_to_filename(output_filename)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  storage_client = storage.Client()
  bucket = storage_client.bucket(BUCKET_NAME)

  # Create output folder
  output_folder = os.path.join(_DATA_DIR.value, FOLDER_NAME)
  os.makedirs(output_folder, exist_ok=True)

  # Download LICENSE file
  copy_blob(bucket, "LICENSE", _DATA_DIR.value)

  # Download data files.
  for filename in FILES:
    blob_name = os.path.join(FOLDER_NAME, filename)
    copy_blob(bucket, blob_name, _DATA_DIR.value)

  print(f"Done downloading data to directory: {_DATA_DIR.value}")


if __name__ == "__main__":
  app.run(main)
