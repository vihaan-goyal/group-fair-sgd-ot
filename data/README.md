# Data

| File | Included | What it is |
|---|---|---|
| `adult.csv` | yes | Adult (Census Income), UCI Machine Learning Repository, CC BY 4.0 |
| `imdb_wiki.csv` | no | IMDb-Wiki metadata: `age, path, split` (e.g. `69, imdb_crop/01/nm0000001_..., train`) |
| `imdb_embeddings.npy` | no | pickled dict `path -> 128-d float face embedding` of the face crops in `imdb_wiki.csv` |

IMDb-Wiki (Rothe, Timofte and Van Gool, IJCV 2018; <https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/>) is released
for non-commercial research only, so the two derived files are not redistributed here. They are available from the
authors on request for research use. Place them in this folder. The runner uses the training split and keeps the
first 100 identities (the `nmXXXXXXX` id in the path) with at least 30 images, 30 images each. SHA-256 of the files
used in the paper:

```
90f9e18dc3f5d91f710193987266c677d5ba08fe3f1d88b3001583781dd57bc1  imdb_wiki.csv
8790e5b03f3f437d2c3e93d483f566db2c5a4fe71e26c16ac2d30765de13c522  imdb_embeddings.npy
```

Only rebuilding the IMDb-Wiki floors (`src/build_tables.py`), rerunning the IMDb-Wiki cells, and the IPFP check need
these files. The committed tables and summaries regenerate every table and figure without them.
