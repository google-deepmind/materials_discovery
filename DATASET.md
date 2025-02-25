# Materials Discovery: GNoME Datasets

*   [GNoME](#gnome)
*   [Convex Hull](#convexhull)
*   [Properties](#properties)
*   [Structures](#structures)

## GNoME <a name="gnome"></a>

The GNoME dataset provides ~381,000 novel structures that update the convex hull
of known stable materials. Associated files are provided in the public Cloud
Bucket defined by `gs://gdm_materials_discovery` with the following
organization:

```
gdm_materials_discovery
└───gnome_data
│   │   stable_materials_summary.csv
│   │   stable_materials_r2scan.csv
│   │   by_composition.zip
│   │   by_id.zip
│   │   by_reduced_formula.zip
│   └───auxiliary_gnome_data
|       |   a2c_supporting_data.json
└───external_data
|   |   mp_snapshot_summary.csv
|   |   external_materials_summary.csv
```

There are two main options for downloading the dataset to a local directory.

*   **Command line:** the command line interface can be used to directly
    download structures from the public bucket via a command of the form `gsutil
    -m cp -r gs://gdm_materials_discovery/ data/`. The `gsutil` command can be
    installed following the
    [Google Cloud CLI install instructions](https://cloud.google.com/sdk/docs/install).
*   **Python script:** Helper scripts have also been provided, which can be run
    using either `scripts/download_data_wget.py` or
    `scripts/download_data_cloud.py`. The latter may be preferable if the user
    already has the Google Cloud CLI authorized.

    First, install the required dependencies. It is best to do this inside a
    virtual environment:

    ```bash
    python -m venv ~/venv/gnome
    source ~/venv/gnome/bin/activate
    pip install absl-py google-cloud-storage
    ```

    The `google-cloud-storage` package is only required for the
    `download_data_cloud.py` script. Then run either:

    ```bash
    python scripts/download_data_wget.py
    ```

    or, after following the
    [Google Cloud CLI install instructions](https://cloud.google.com/sdk/docs/install),

    ```bash
    gcloud auth application-default login
    python scripts/download_data_gcloud.py
    ```

    Both scripts take an optional flag, `--data_dir`, to control the directory
    the data is downloaded to. By default, the `data` directory in the current
    working directory is used, and created if required.

## Convex Hull <a name="convexhull"></a>

First, in `stable_materials_summary.csv`, we provide a CSV containing the
compositions of all novel materials as well as corresponding energies (along
with a number of other properties). This file is the simplest and most useful
for calculating the convex hull energies over all stable materials. New
materials can be benchmarked against this file for decomposition energy
estimates. We have updated this csv to ensure minimal overlap with newer
versions of external datasets such as OQMD and MP, based on snapshots in June
2024.

Note, construction of the complete convex hull requires energies from Materials
Project (MP), the Open Quantum Materials Database (OQMD), and WBM. We have
included `external_materials_summary.csv` to provide composition and associated
convex hull entries with elemental compositions that match these external
datasets, though in some cases the improved energies correspond to lower energy
structures. To ensure compatibility between DFT calculations, we correct the
energies of all entries in the Materials Project that contain any of Ga, Ge, Li,
Mg, Na, elements for which the pseudopotentials we use and the ones used by MP
differ. The correction per atom is provided below:

```python:
  pp_corr = {"Ga": -0.0028805, "Ge": 0.10417085, "Li": -0.00301278, "Mg": 0.0924014, "Na": -0.00447437}
```

The combination of these two datasets provides the updated convex hull and can
be used for evaluating the stability of other computational experiments. In
additional, for the example colabs, we also provide a snapshot of the stable
crystals from the Materials Project, enabling visualization of the exploration
spaces of GNoME.

## Properties <a name="properties"></a>

Beyond energy, a number of additional properties may be of interest for the
novel crystal structures. `stable_materials_summary.csv` provides a number of
additional descriptors that may be helpful when processing or looking for
crystals with specific properties.

A list of the properties provided and textual descriptions is provided below:

*   **Composition:** alphabetically-ordered composition
*   **MaterialId:** a unique id corresponding to the entry
*   **Reduced Formula:** reduced chemical formula
*   **Elements:** chemical system
*   **NSites:** number of atoms
*   **Volume:** volume in units Å^3
*   **Density:** density in units Å^3 / atom
*   **Point Group:** assigned point group
*   **Space Group:** assigned space group
*   **Space Group Number:** assigned space group number
*   **Crystal System:** assigned crystal system
*   **UnCorrected Energy:** uncorrected energy
*   **Corrected Energy:** energy adjusted by MP2020 corrections
*   **Formation Energy Per Atom:** normalized energy corrected by reference
    elements
*   **Decomposition Energy Per Atom:** decomposition energy relative to the
    downloaded Materials Project convex hull
*   **Dimensionality Cheon:** dimensionality predicted by Cheon et al. 2017
*   **Bandgap:** calculated bandgap (if available)
*   **Is Train:** in training set for associated machine learning models
*   **Decomposition Energy Per Atom All:** distance to convex hull of all
    entries
*   **Decomposition Energy Per Atom Relative:** distance to convex hull of all
    entries except for the current
*   **Decomposition Energy Per Atom MP:** distance to convex hull of all entries
    from Materials Project
*   **Decomposition Energy Per Atom MP OQMD:** distance to convex hull of all
    entries from Materials Project + Open Quantum Materials Database (including
    recalculations)

Formation energies per atom are all calculated with respect to a fixed set of
elemental references from the Materials Project (and not lowered if we found a
lower) energy elemental structure. For clarity, we have provided a list of the
elemental references used in JSON form in `elemental_references.json`

## Structures <a name="structures"></a>

Each of the associated rows of the CSV provided above is associated with a
crystal structure. We provide compressed directories of associated CIFs, a
standard file format within the materials science community. Note, three
versions of the compressed directories exist, where file names allow for lookup
by unique identifier, reduced formula, or by composition.

## r²SCAN

Validation of the associated structures was completed using r²SCAN.
`stable_materials_r2scan.csv` provides all r²SCAN calculations performed on
stable materials. Note, stability metrics change with the choice functional (as
discussed in the associated paper), so not all released materials remain stable
according to this metric.

## Caveats

Due to numerical precision (and errors arising from the computational
simulations), we use a threshold of 5e-5 eV as the threshold for determining
whether a material is on the convex hull. For all measurements, the provided
materials update the convex hull of a snapshotted version of Materials Project
and similar databases. Therefore, as more crystals are discovered by the
scientific community, the above set may not remain stable.

## Versioning

Below, we keeps notes about any upgrades made to the dataset as well as
approximate timing.

*   (11/29/23) Initial dataset release
*   (12/1/23) Re-introduce paper filters to remove un-physical energies; add 2
    missing columns ('Dimensionality Cheon' and 'Is Train')
*   (8/21/24) Adjust threshold to 1meV/atom off the hull significantly
    increasing the number of released crystals, update to use "require_bound"
    version of MP corrections, re-relax structure where VASP did not update
    lattice coordinates, provide convex hull entries from recomputations of MP /
    OQMD / ..., provided colabs for accessing, fixed file naming in by_id.zip
*   (11/21/24) Ensure consistency in formation energies, provide used elemental
    references + pseudopotential corrections used in the GNoME paper

## Disclaimer

Additional data that are not as part of the core GNoME effort (such as e.g. band
gaps) are provided as-is, at PBE-level, are work in progress, and may be updated
over time. Continued efforts to characterize the electronic properties of stable
materials may update / correct these values. More accurate calculations and
processing are a work-in-progress. Other auxiliary data related to GNoME efforts
(e.g. structure prediction with [a2c](https://arxiv.org/abs/2310.01117)) is
available under auxiliary_gnome_data.
