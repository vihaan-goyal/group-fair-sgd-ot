# Data

| File | What it is |
|---|---|
| `adult.csv` | Adult (Census Income), UCI Machine Learning Repository, CC BY 4.0 |
| `imdb_wiki.csv` | IMDb-Wiki metadata: `age, path, split` (e.g. `69, imdb_crop/01/nm0000001_..., train`) |
| `imdb_embeddings.npy` | pickled dict `path -> 128-d float face embedding` of the face crops in `imdb_wiki.csv` |

The two IMDb-Wiki files are derived from IMDb-Wiki (Rothe, Timofte and Van Gool, IJCV 2018;
<https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/>), which is released for non-commercial research only. They are
included here for reproducing the paper and fall under the same terms. The runner uses the training split and
keeps the first 100 identities (the `nmXXXXXXX` id in the path) with at least 30 images, 30 images each.

SHA-256:

```
90f9e18dc3f5d91f710193987266c677d5ba08fe3f1d88b3001583781dd57bc1  imdb_wiki.csv
8790e5b03f3f437d2c3e93d483f566db2c5a4fe71e26c16ac2d30765de13c522  imdb_embeddings.npy
```
