# gps_cluster

Clustering of GPS-derived velocities using Hierarchical Agglomerative
Clustering (HAC, e.g. Simpson et al., 2012). HAC begins with each observation
being its own cluster, then sequentially combines similar observations or
clusters until all the observations are grouped into a single cluster. Each
merging can be represented with a dendrogram. This is essentially a linkage
graph with inter-cluster distances in the y-axis. Then using the dendrogram, gap
statistics and a priori deformation indicators optimal number of clusters is
determined.

## Description

The clustering of GPS-derived velocity vectors provides an objective and intuitive starting point for the analysis of block-like deformation and yields
a statistically significant number of blocks. Cluster analysis is not a new
method, it has applications in various discipline, such as psychology,
bioinformatics, and data analysis, but the application is rather recent for
continental deformation (Simpson et al., 2012; Savage and Simpson, 2013a,b; Liu
et al., 2015; Savage and Wells, 2015; Thatcher et al., 2016; Savage, 2018;
Takahashi et al., 2019; Özdemir and Karslıoğlu, 2019).

There are two popular clustering algorithms: k-means and Hierarchical
Agglomerative Clustering (HAC). k-means finds cluster centers for a given
number of clusters, minimizing the within-cluster dispersion. Its easy
implementation, linear execution time scaling with the number of observations,
and guaranteed convergence renders it one of the most popular clustering
algorithms. k-means perform poorly when clusters are of varying sizes.
Furthermore, clustering is sensitive to data outliers and the initial sorting
of the data. Therefore, pre-processing and repeat experiments need to be done.

HAC builds a hierarchy of clusters without having a fixed number of clusters.
HAC begins with each observation being its own cluster, then sequentially joins
near observations (or clusters) until all the observations are grouped into a
single cluster. As pairwise distances are calculated at each step, the
algorithm is slower compared to k-means and scales with O(n2). However,
”agglomerative clustering can provide a whole hierarchy of possible partitions
of the data, which can be easily inspected via dendrograms.” (M ̈uller and
Guido, 2016). This is essentially a linkage graph with inter-cluster
distances in the y-axis. Then inspection of the dendrogram or variance-
reduction measures give an indication for the optimal number of clusters. HAC
methods work best with low dimensional data.

Simpson et al. (2012) applied a hierarchical agglomerative clustering approach
to GPS velocity vectors in the San Francisco Bay region. The resulting clusters
and geologically determined fault-bounded blocks are in good agreement. As the
displacements in the Bay Area are predominantly translational, station
locations are not taken into account.

Savage and Simpson (2013a,b) applied k-medoids clustering and improved the
method by considering station locations on the clustering. After initial
clustering of velocities, Euler vectors for each cluster are calculated. Then
iteratively, each observation is assigned to that cluster, for which its Euler
vector best describes

In this study we used the HAC algorithm.

## Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an
   environment `gps_cluster` with the help of [conda]: ``` conda env create -f
environment.yml ```
2. activate the new environment with: ``` conda activate gps_cluster ```

> **_NOTE:_**  The conda environment will have gps_cluster installed in
> editable mode.  Some changes, e.g. in `setup.cfg`, might require you to run
> `pip install -e .` again.


Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with: ```bash pre-commit install #
You might also want to run `pre-commit autoupdate` ``` and checkout the
configuration under `.pre-commit-config.yaml`.  The `-n, --no-verify` flag of
`git commit` can be used to deactivate pre-commit hooks temporarily.

4. install [nbstripout] git hooks to remove the output cells of committed
notebooks with: ```bash nbstripout --install --attributes
notebooks/.gitattributes ``` This is useful to avoid large diffs due to plots
in your notebooks.  A simple `nbstripout --uninstall` will revert these
changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in
`environment.yml` and eventually in `setup.cfg` if you want to ship and install
your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact
reproduction of your environment with: ```bash conda env export -n gps_cluster
-f environment.lock.yml ``` For multi-OS development, consider using
`--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml`
using: ```bash conda env update -f environment.lock.yml --prune ``` ## Project
Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build system configuration. Do not change!
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual Python package, e.g. train_model.py.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `pip install -e .` to install for development or
|                              or create a distribution with `tox -e build`.
├── src
│   └── gps_cluster         <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.1 and the [dsproject extension] 0.6.1.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
