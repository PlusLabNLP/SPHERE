# ENERGY-REGULARIZED SEQUENTIAL MODEL EDITING ON HYPERSPHERES (SPHERE)


![alt text](resource/sphere_sparse.png)
*Figure: a) A weight matrix is viewed as a set of neurons (red dots) on a hypersphere. (b) Current SOTA methods introduce perturbations (blue triangles) that
interfere with the principle hyperspherical directions of pre-edit weights. (c) SPHERE projects new knowledge onto a sparse space complementary to the principal hyperspherical directions.*


## NEWS
- 🔥 [2025.9.29] SPHERE is released.

## Requirements
- pytorch==1.12.1
- einops==0.4.0
- higher==0.2.1
- hydra-core==1.2.0
- transformers==4.30.1
- datasets==1.18.3
- matplotlib==3.6.1
- spacy==3.4.1
- scipy==1.9.2
- scikit-learn==1.0.2
- nltk==3.7


## Quick Start
### An example for editing Qwen2.5 (7B) on counterfact dataset using SPHERE
#### 1. Edit Qwen2.5 (7B) model 
 
    python3 -m experiments.evaluate     --alg_name=AlphaEdit     --model_name=./Qwen2.5-7B-Instruct     --hparams_fname=Qwen2.5-7B.json --ds_name=mcf --dataset_size_limit=5000    --num_edits=100 --beta_hse=0.5 --alpha=0.5

This command runs an evaluation script for the SPHERE algorithm using the Qwen2.5-7B-Instruct. Below are the explanations for each argument:

- `--alg_name=AlphaEdit`: Specifies the name of the algorithm being used, which is AlphaEdit in this case.
- `--model_name=./Qwen2.5-7B-Instruct`: Indicates the name of the model being evaluated, here it is Qwen2.5-7B-Instruct.
- `--hparams_fname=Qwen2.5-7B.json`: Points to the JSON file containing hyperparameters specific to the Qwen2.5-7B-Instruct model.
- `--ds_name=mcf`: Specifies the dataset name, in this case, "mcf".
- `--dataset_size_limit=5000`: Sets the total number of editing samples to 5000.
- `--num_edits=100`: Defines the batch size for each round of editing, meaning 100 edits will be performed in each batch. 
- `--beta_hse=0.5`: Cumulative Ratio, meaning that only the top 50% of the principal directions of the edited weights are suppressed.
- `--alpha=0.5`: Suppression Strength, define the suppression strength in the projection, which controls the extent to which perturbation components along the principal directions are removed.

If you want to run the baseline, please set beta_hse to 0.
If you want to use SPHERE on MEMIT/PRUNE/RECT, please set beta_hse=0.5, alpha=0.8 to reproduce the paper result.

The edited weight (selected) from each run are stored at `./Edited_Weight/<alg_name>/<model_name>/<dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>/` in a specific format:
```bash
.Edited_Weight/
    |__ <alg_name>/
        |__ <model_name>/
            |__ <dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>
            |__ <dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>
            |__ ...
            |__ <dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>
```

#### 2. Editing Evaluation
 
    python3 -m scripts.evaluate_each_epoch    --model_name=./Qwen2.5-7B-Instruct     --weight_folder=./Edited_Weight/<alg_name>/<model_name>/<dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>/ --ds_name=mcf --dataset_size_limit=5000  --generation_test_interval=100

This command runs an evaluation script for the SPHERE algorithm using the Qwen2.5-7B-Instruct. Below are the explanations for each argument:

- `--model_name=./Qwen2.5-7B-Instruct`: Indicates the name of the model being evaluated, here it is Qwen2.5-7B-Instruct.
- `--weight_folder`: Points to the saved weight from previous editing.
- `--ds_name=mcf`: Specifies the dataset name, in this case, "mcf".
- `--dataset_size_limit=5000`: Sets the total number of evaluation samples to (first) 5000.
- `--generation_test_interval=100`: indicates that a test generation is conducted after every 100 rounds of evaluation.

The evaluation results will be stored at the `./Edited_Weight/<alg_name>/<model_name>/<dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>/summary/summary.json`

#### 2. Downstream Tasks Evaluation
 
    python3 -m scripts.evaluate_each_epoch    --model_name=./Qwen2.5-7B-Instruct     --weight_folder=./Edited_Weight/<alg_name>/<model_name>/<dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>/

This command runs an evaluation script for the SPHERE algorithm using the Qwen2.5-7B-Instruct. Below are the explanations for each argument:

- `--model_name=./Qwen2.5-7B-Instruct`: Indicates the name of the model being evaluated, here it is Qwen2.5-7B-Instruct.
- `--weight_folder`: Points to the saved weight from previous editing.

The evaluation results will be stored at the `./Edited_Weight/<alg_name>/<model_name>/<dataset>_weight_data_batch_<batch_size>_<beta_hse>_<alpha>/rect_eval/`


## Citation
If you use this code and dataset, please cite our paper:
```bibtex
@article{liu2025energy,
  title={Energy-Regularized Sequential Model Editing on Hyperspheres},
  author={Liu, Qingyuan and Gu, Jia-Chen and Yao, Yunzhi and Wang, Hong and Peng, Nanyun},
  journal={arXiv preprint arXiv:2510.xxxxx},
  year={2025}
}
```



## Acknowledgment
Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git), [``EMMET``](https://github.com/scalable-model-editing/unified-model-editing.git) and [``AlphaEdit``](https://github.com/jianghoucheng/AlphaEdit.git).
