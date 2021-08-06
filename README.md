# `leiden_clustering`
[![pipy](https://img.shields.io/pypi/v/leiden_clustering)](https://pypi.python.org/pypi/leiden_clustering)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Description
Class wrapper based on [`scanpy`](https://scanpy.readthedocs.io/en/stable/) to use the Leiden algorithm to directly cluster your data matrix with a `scikit-learn` flavor.

## Requirements
Developed using:
- `scanpy` v1.7.2
- `sklearn` v0.23.2
- `umap` v0.4.6
- `numpy` v1.19.2
- `leidenalg`

## Installation
### pip
```shell
pip install leiden_clustering
```
### local
```shell
git clone https://github.com/MiqG/leiden_clustering.git
cd leiden_clustering
pip install -e .
```

## Usage
```python
from leiden_clustering import LeidenClustering
import numpy as np
X = np.random.randn(100,10)
clustering = LeidenClustering()
clustering.fit(X)
clustering.labels_
```

## License
`leiden_clsutering` is distributed under a BSD 3-Clause License (see [LICENSE](https://github.com/CRG-CNAG/leiden_clustering/blob/main/LICENSE)).

## References
- *Traag, V.A., Waltman, L. & van Eck, N.J.* From Louvain to Leiden: guaranteeing well-connected communities. Sci Rep 9, 5233 (2019). DOI: https://doi.org/10.1038/s41598-019-41695-z
