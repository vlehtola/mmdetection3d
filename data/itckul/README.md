### Prepare itckul Data

1. Download itckul data. Link or move the folder to this level of directory.

2. In this directory, extract point clouds and annotations by running `python collect_indoor3d_data.py`.

3. Enter the project root directory, generate training data by running

```bash
python tools/create_data.py itckul --root-path ./data/itckul --out-dir ./data/itckul --extra-tag itckul
```

The overall process could be achieved through the following script

```bash
python collect_indoor3d_data.py
cd ../..
python tools/create_data.py itckul --root-path ./data/itckul --out-dir ./data/itckul --extra-tag itckul
```

The directory structure after pre-processing should be as below

```
itckul
├── meta_data
├── indoor3d_util.py
├── collect_indoor3d_data.py
├── README.md
├── itckul_data
TODO

```