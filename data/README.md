# Data

| File | What it is |
|---|---|
| `adult.csv` | Adult (Census Income), UCI Machine Learning Repository, CC BY 4.0 |
| `imdb_wiki.csv` | IMDb-Wiki metadata: `age, path, split` (e.g. `69, imdb_crop/01/nm0000001_..., train`) |
| `imdb_embeddings.npy` | pickled dict `path -> 128-d float face embedding` of the face crops in `imdb_wiki.csv` |

The two IMDb-Wiki files are derived from IMDb-Wiki (Rothe, Timofte and Van Gool, IJCV 2018;
<https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/>), which is released for non-commercial research only. They are
included here for reproducing the paper and fall under the same terms.

- `imdb_wiki.csv` is the IMDB-WIKI-DIR metadata file of Yang et al., "Delving into Deep Imbalanced Regression"
  (ICML 2021), copied unchanged from <https://github.com/YyzHarry/imbalanced-regression/tree/main/imdb-wiki-dir>
  (MIT License, notice below).
- `imdb_embeddings.npy` holds, for the face crops the experiments use, ResNet18 penultimate-layer features (512-d),
  reduced to 128 dimensions by PCA and standardized.

The runner uses the training split and keeps the first 100 identities (the `nmXXXXXXX` id in the path) with at least
30 images, 30 images each.

SHA-256:

```
3bae6010275e967defaeb0cce60b854c888d95e00cf52e1df33d3fd85be69ebd  imdb_wiki.csv   (LF line endings; a Windows checkout with autocrlf gives CRLF)
8790e5b03f3f437d2c3e93d483f566db2c5a4fe71e26c16ac2d30765de13c522  imdb_embeddings.npy
```

## License notice for `imdb_wiki.csv`

```
MIT License

Copyright (c) 2021 Yuzhe Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
