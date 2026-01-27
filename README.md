# DRIP — Data-Reward Integrated Planner for Residential Floorplan Generation


## Repository structure

- `main.py` — training / full-pipeline entry (class: `DQN` in `dqn.py`)
- `inference.py` — inference engine (class: `InferenceEngine`)
- `env_fixed_state_length.py` — environment / layout generator (class: `LayoutGenerator`)
- `model.py` — neural network model (class: `DQN_Agent_Transformer_GAT_PRE`)
- `dqn.py` — RL agent implementation and training loop
- `dataset/` — place evaluation / test datasets here
- `base_model/` — small pretrained base models used by the environment
- `param/` — trained model parameter files
- `logs/` — logs are written here during runs
- `human_voting_data/` — human evaluation data and results

## Requirements

- Python 3.8+
- PyTorch (matching your CUDA if using GPU)
- Other Python packages listed in `requirements.txt`

## Setup
1. Create and activate a new Python environment using conda:
```sh
conda create -n drl_env python=3.12
```
2. Install pytorch. Example for CUDA 12.4:
```sh
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
3. Install other dependencies:
```sh
pip install -r requirements.txt
```

Note: torch-geometric installation may require additional platform-specific steps. See the official instructions at https://pytorch-geometric.readthedocs.io.

## Quick start

All commands assume you are inside the repository root on a Windows machine and have a Python environment activated.

1. Training
- Edit configuration in `main.py` or pass your own config before running. Important: adjust `worker_num` in `main.py` to match your CPU/GPU parallel capabilities. Larger `worker_num` increases parallelism but also memory/CPU usage.
- Run training:
```sh
python main.py
```
Training will create log folders under the configured `log_dir` and save model parameters under the configured parameter path (see `CONFIG` in `main.py`).

2. Inference
- By default `inference.py` uses a model path and base model path set near the bottom of the file. Update these variables to point to your trained model (`model_path`) and the `base_model` folder if needed.
- Run inference:
```sh
python inference.py
```
- Logs and per-instance error logs are written under `./logs/<timestamp>/`.

Note: Inference engine is single-threaded, so it may take longer on large datasets. If you need faster inference, consider running `main.py` by commonting out the training line and uncommenting the prediction line.

## Using custom datasets

To evaluate the model on your own datasets:

1. Create a new dataset file in the `dataset/` folder. The code expects either `.xlsx` or `.csv` files.
2. If you have a precomputed graph file for a dataset, name it `<basename>_graph.xlsx` and place it in the same dataset folder.

Example:
- `dataset/my_test_floor.xlsx`
- Optional graph file: `dataset/my_test_floor_graph.xlsx`

## Configuration tips

- main.py `CONFIG`:
  - `worker_num`: number of parallel workers for environment simulation. Increase on systems with more CPU cores and sufficient memory.
  - `lr_max`, `lr_min`, `batch_size`, etc.: hyperparameters for training.
  - `log_dir` and `para_path`: control where logs and parameters are saved.

- inference.py:
  - `model_path`: path to the saved model weights (change to your trained `.pth` file).
  - `base_model_path`: path to the `base_model` folder used by the environment.

## Logging & outputs

- Training and inference print progress information to the console and write detailed logs under the `logs/` directory.
- Per-instance inference errors are saved to `logs/<timestamp>/error_log/<file>_output.log`.

## Reproducibility

- See `main.py` for a `set_seed()` helper used to fix random seeds (PyTorch, numpy, random).
- For deterministic behavior on CUDA, you may need to set additional PyTorch backend flags (see PyTorch docs).

## Troubleshooting

- CUDA / GPU issues: ensure PyTorch and CUDA versions are compatible.
- torch-geometric install errors: follow the platform-specific install instructions on the torch-geometric site.
- If `inference.py` reports "no initial feasible solution" for a floor, check the dataset formatting and any required base model assets.

## License & citation

See LICENSE file for details.

This repository contains code for training and inference, with a minimal dataset to reproduce key results. Full dataset and training pipeline will be released after further IP protection.

## Contact

For issues, please open an issue in this repository with reproducible steps and the relevant log files found under `logs/`.