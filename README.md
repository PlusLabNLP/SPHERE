<div align="center">


# 🌐 SPHERE

## ICLR 2026: Energy-Regularized Sequential Model Editing on Hyperspheres
<h3><em>Still one line of code. Boom your editing performance!</em></h3>

<p><em>If this project helps you, a star ⭐ would mean a lot to us. </em>😊😊</p>

[![arXiv](https://img.shields.io/badge/arXiv-2510.01172-b31b1b.svg)](https://arxiv.org/abs/2510.01172)
[![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.2510.01172.svg)](https://doi.org/10.48550/arXiv.2510.01172)
[![Venue](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/virtual/2026/poster/10010872)

<p align="center">
  <a href="#-installation">📦 Installation</a> •
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="https://www.qingyuanliu.net/sphere_projectpage/">🌐 Project Page</a> •
  <a href="https://github.com/zjunlp/EasyEdit/tree/main/easyeditor/models/SPHERE" target="_blank">✏️ EasyEdit</a> •
  <a href="https://arxiv.org/abs/2510.01172">📄 Paper</a> •
  <a href="https://tianzhaohaha.github.io/uploads/NICE_Slide_Sim.pdf">📊 Slides</a> •
  <a href="#" target="_blank">🎬 Video</a>
</p>

</div>

---

<p align="center">
  <img src="resource/sphere_sparse.png" width="95%" alt="SPHERE Overview"/>
</p>

<p align="center"><em>
<b>Figure:</b> (a) A weight matrix is viewed as a set of neurons (red dots) on a hypersphere.
(b) Current SOTA methods introduce perturbations (blue triangles) that interfere with the principal hyperspherical directions of pre-edit weights.
(c) SPHERE projects new knowledge onto a sparse space complementary to the principal hyperspherical directions.
</em></p>



## 📰 News

- 🔥 **[2026.03]** We release the pre-computed **cov** matrices for quick reproduction. See [Download](#-download).
- 🔥 **[2026.02]** SPHERE is supported in [EasyEdit](https://github.com/zjunlp/EasyEdit/tree/main/easyeditor/models/SPHERE).
- 🎉 **[2026.01]** SPHERE is accepted by **ICLR 2026** (Score: 8884, Top-1.1% in Transfer/Meta/Lifelong Learning track).
- 🚀 **[2025.09]** SPHERE is released.


## 📦 Installation

```bash
pip install torch==1.12.1
pip install einops==0.4.0 higher==0.2.1 hydra-core==1.2.0
pip install transformers==4.30.1 datasets==1.18.3
pip install matplotlib==3.6.1 spacy==3.4.1
pip install scipy==1.9.2 scikit-learn==1.0.2 nltk==3.7
```

<details>
<summary>📋 Full dependency list</summary>

| Package | Version |
|:---|:---|
| pytorch | 1.12.1 |
| einops | 0.4.0 |
| higher | 0.2.1 |
| hydra-core | 1.2.0 |
| transformers | 4.30.1 |
| datasets | 1.18.3 |
| matplotlib | 3.6.1 |
| spacy | 3.4.1 |
| scipy | 1.9.2 |
| scikit-learn | 1.0.2 |
| nltk | 3.7 |

</details>


## 📥 Download

We provide the pre-computed **cov** matrix for both **Llama3-8B-Instruct** and **Qwen2.5-7B-Instruct** via [Google Drive](https://drive.google.com/drive/folders/17Ea1yxQcnfdhQUR43EWhfJImGxmX4QJh?usp=sharing).

After downloading, decompress the file and place it under the `./data/stats` directory.


## 🚀 Quick Start

> **Example:** Editing Qwen2.5 (7B) on the CounterFact dataset using SPHERE

### Step 1: Edit the Model

```bash
python3 -m experiments.evaluate \
    --alg_name=AlphaEdit \
    --model_name=./Qwen2.5-7B-Instruct \
    --hparams_fname=Qwen2.5-7B.json \
    --ds_name=mcf \
    --dataset_size_limit=5000 \
    --num_edits=100 \
    --beta_hse=0.5 \
    --alpha=0.5
```

<details>
<summary>🔧 <b>Argument details</b></summary>

| Argument | Description |
|:---|:---|
| `--alg_name` | Algorithm name (e.g., `AlphaEdit`) |
| `--model_name` | Path to the model (e.g., `./Qwen2.5-7B-Instruct`) |
| `--hparams_fname` | Hyperparameter JSON file (e.g., `Qwen2.5-7B.json`) |
| `--ds_name` | Dataset name (e.g., `mcf`) |
| `--dataset_size_limit` | Total number of editing samples |
| `--num_edits` | Batch size for each round of editing |
| `--beta_hse` | **Cumulative Ratio** — top percentage of principal directions to suppress (e.g., `0.5` = top 50%) |
| `--alpha` | **Suppression Strength** — controls extent of perturbation removal along principal directions |

</details>

> [!TIP]
> - To run the **baseline**, set `beta_hse=0`.
> - To use SPHERE on **MEMIT / PRUNE / RECT**, set `beta_hse=0.5, alpha=0.8` to reproduce paper results.

The edited weights from each run are stored as:

```
📂 Edited_Weight/
└── 📂 <alg_name>/
    └── 📂 <model_name>/
        ├── 📁 <dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>/
        ├── 📁 <dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>/
        └── ...
```

### Step 2: Editing Evaluation

```bash
python3 -m scripts.evaluate_each_epoch \
    --model_name=./Qwen2.5-7B-Instruct \
    --weight_folder=./Edited_Weight/<alg_name>/<model_name>/<dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>/ \
    --ds_name=mcf \
    --dataset_size_limit=5000 \
    --generation_test_interval=100
```

<details>
<summary>🔧 <b>Argument details</b></summary>

| Argument | Description |
|:---|:---|
| `--model_name` | Path to the model being evaluated |
| `--weight_folder` | Path to saved weights from previous editing |
| `--ds_name` | Dataset name (e.g., `mcf`) |
| `--dataset_size_limit` | Total number of evaluation samples |
| `--generation_test_interval` | Run test generation every N evaluation rounds |

</details>

📊 Results are saved to:
`./Edited_Weight/<alg_name>/<model_name>/<dataset>_weight_data_batch_<...>/summary/summary.json`

### Step 3: Downstream Tasks Evaluation

```bash
python3 -m scripts.evaluate_each_epoch \
    --model_name=./Qwen2.5-7B-Instruct \
    --weight_folder=./Edited_Weight/<alg_name>/<model_name>/<dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>/
```

📊 Results are saved to:
`./Edited_Weight/<alg_name>/<model_name>/<dataset>_weight_data_batch_<...>/rect_eval/`


## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{liu2026energy,
  title     = {Energy-Regularized Sequential Model Editing on Hyperspheres},
  author    = {Liu, Qingyuan and Gu, Jia-Chen and Yao, Yunzhi and Wang, Hong and Peng, Nanyun},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026}
}
```


## 🙏 Acknowledgment & Contact

Our code is built upon [**MEMIT**](https://github.com/kmeng01/memit.git), [**EMMET**](https://github.com/scalable-model-editing/unified-model-editing.git), and [**AlphaEdit**](https://github.com/jianghoucheng/AlphaEdit.git). If you have any questions, feel free to reach out at **ql2505(at)columbia.edu**.
