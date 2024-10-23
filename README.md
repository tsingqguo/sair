# SAIR ECCV2024
We propose the semantic-aware implicit representation by learning semantic-aware implicit representation (SAIR), that is, we make the implicit representation of each pixel rely on both its appearance and semantic information (e.g., which object does the pixel belong to). This work is publised in ECCV 2024.

### Environment
- Python 3
- Pytorch 1.6.0
- TensorboardX
- yaml, numpy, tqdm, imageio


### Data

`mkdir load` for putting the dataset folders.

- **celebAHQ**: `mkdir load/celebAHQ` and `cp scripts/resize.py load/celebAHQ/`, then `cd load/celebAHQ/`. Download and `unzip` data1024x1024.zip from the [Google Drive link](https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P?usp=sharing) (provided by [this repo](github.com/suvojit-0x55aa/celebA-HQ-dataset-download)). Run `python resize.py` and get image folders `256/, 128/, 64/, 32/`. Download the [split.json](https://www.dropbox.com/s/2qeijojdjzvp3b9/split.json?dl=0).

### Running the code

**0. Preliminaries**

- For `train_liif.py` or `test.py`, use `--gpu [GPU]` to specify the GPUs (e.g. `--gpu 0` or `--gpu 0,1`).

- For `train_liif.py`, by default, the save folder is at `save/_[CONFIG_NAME]`. We can use `--name` to specify a name if needed.


**1. celebAHQ experiments**

**Train**: `python train_liif.py --config configs/train-celebAHQ/[CONFIG_NAME].yaml`.

**Test**: `python test.py --config configs/test/test-celebAHQ-32-256.yaml --model [MODEL_PATH]` (or `test-celebAHQ-64-128.yaml` for another task). We use `epoch-best.pth` in corresponding save folder.

## Bibtex

```

@inproceedings{zhang2025sair,
  title={Sair: Learning semantic-aware implicit representation},
  author={Zhang, Canyu and Li, Xiaoguang and Guo, Qing and Wang, Song},
  booktitle={European Conference on Computer Vision},
  pages={319--335},
  year={2025},
  organization={Springer}
}

```
