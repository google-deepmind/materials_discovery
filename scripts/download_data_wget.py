# Copyright 2023 DeepMind Technologies Limited
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

flags.DEFINE_string("data_dir", "data", "Location to copy downloaded data.")

FLAGS = flags.FLAGS

PUBLIC_LINK = "https://storage.googleapis.com/"
BUCKET_NAME = "gdm_materials_discovery"
FOLDER_NAME = "gnome_data"
FILES = [
    "stable_materials_hull.csv",
    "stable_materials_r2scan.csv",
    "stable_materials_summary.csv",
    "by_composition.zip",
    "by_id.zip",
    "by_reduced_formula.zip",
]


def download_from_link(link, output_dir):
  """Download a file from a public link using wget."""
  os.system(f"wget {link} -P {output_dir}")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Create output folder
  output_folder = os.path.join(FLAGS.data_dir, FOLDER_NAME)
  os.makedirs(output_folder, exist_ok=True)

  parent_directory = os.path.join(PUBLIC_LINK, BUCKET_NAME, FOLDER_NAME)

  # Download LICENSE file
  download_from_link(
      os.path.join(PUBLIC_LINK, BUCKET_NAME, "LICENSE"), FLAGS.data_dir
  )

  for filename in FILES:
    public_link = os.path.join(parent_directory, filename)
    download_from_link(public_link, os.path.join(FLAGS.data_dir, FOLDER_NAME))

  print(f"Done downloading data to directory: {FLAGS.data_dir}")


if __name__ == "__main__":
  app.run(main)
