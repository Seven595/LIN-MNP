# Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching

This is our PyTorch implementation for the paper:

> Tinglin Huang, Tianyu Liu, Mehrtash Babadi, Wengong Jin, and Rex Ying (2025). Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching. Paper in [arxiv](https://arxiv.org/pdf/2506.05361).

We recently extended STFlow and released **STPath**, a generative pretrained model capable of directly predicting the expression levels of 38,984 genes from histology images without further fine-tuning. Feel free to check out the [paper](https://www.biorxiv.org/content/10.1101/2025.04.19.649665v2.abstract) and [code](https://github.com/Graph-and-Geometric-Learning/STPath), which includes an easy-to-use API.

## Organization

The organization of this repository is as follows:
- `app/`: contains the training pipelines for pathology foundation models and STFlow
    - `hest/`: training pipeline for pathology foundation models, mainly from HEST pipeline
    - `flow/`: training pipeline for STFlow
- `data/`: contains the dataloader for the STFlow model
- `model/`: contains the implementation of denoiser
- `hest_utils/`: contains the utility functions for pathology foundation models, mainly from HEST pipeline


## Usage

Please run install the package by running:

```
$ pip install -e .
```

Download HEST benchmark datasets and pretrained weights of UNI and GigaPath models using the following script:
```
from huggingface_hub import snapshot_download, hf_hub_download

source_dataroot = "./data"
weights_root = "./models"

snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=weights_root, allow_patterns=['fm_v1/*'])
snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=source_dataroot, ignore_patterns=['fm_v1/*'])
hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "uni/"))
hf_hub_download("prov-gigapath/prov-gigapath", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "gigapath/"))
```

Testing foundation models with the following script, which will save the extracted features in the `embed_dataroot`:
```
$ python app/hest/benchmark.py \
        --datasets all \
        --encoders uni_v1_official \
        --weights_root /path/to/weights_root \
        --source_dataroot /path/to/source_dataroot \
        --embed_dataroot /path/to/embed_dataroot \
        --batch_size 128
```

Training STFlow with the following script:
```
$ python app/flow/train.py \
        --datasets LUNGS \
        --feature_encoder uni_v1_official \
        --source_dataroot /path/to/source_dataroot \
        --embed_dataroot /path/to/embed_dataroot \
        --batch_size 2 \
        --n_layers 4 \
        --n_sample_steps 5
```

## Reference

If you find our work useful in your research, please consider citing our paper:

```
@inproceedings{huang2025stflow,
  title={Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching},
  author={Huang, Tinglin and Liu, Tianyu and Babadi, Mehrtash and Jin, Wengong and Ying, Rex},
  booktitle={International Conference on Machine Learning},
  year={2025}
}

@article{huang2025stpath,
  title={STPath: A Generative Foundation Model for Integrating Spatial Transcriptomics and Whole Slide Images},
  author={Huang, Tinglin and Liu, Tianyu and Babadi, Mehrtash and Ying, Rex and Jin, Wengong},
  journal={bioRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```
