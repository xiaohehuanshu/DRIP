# DRIP — Data-Reward Integrated Planner for Residential Floorplan Generation


## Repository structure

- `main.py` — training / full-pipeline entry (class: `DQN` in `dqn.py`)
- `inference.py` — inference engine (class: `InferenceEngine`)
- `env_fixed_state_length.py` — environment / layout generator (class: `LayoutGenerator`)
- `model.py` — neural network model (class: `DQN_Agent_Transformer_GAT_PRE`)
- `dqn.py` — RL agent implementation and training loop
- `dataset/` — place training / evaluation datasets here
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
2. Install pytorch. See https://pytorch.org/get-started/previous-versions/ for the correct command based on your system and CUDA version.
Example for CUDA 12.4:
```sh
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
or for CPU-only:
```sh
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```
3. Install other dependencies:
```sh
pip install -r requirements.txt
```

Note: torch-geometric installation may require additional platform-specific steps. See the official instructions at https://pytorch-geometric.readthedocs.io.

## Quick start

All commands assume you are inside the repository root on a Windows/Linux machine and have a Python environment activated.

1. Training
- Edit configuration in `main.py` or pass your own config before running.

  Important: adjust `worker_num` in `main.py` to match your CPU/GPU parallel capabilities. Larger `worker_num` increases parallelism but also memory/CPU usage.
- Run training:
```sh
python main.py
```
Training will create log folders under the configured `log_dir` and save model parameters under the configured parameter path (see `CONFIG` in `main.py`).

2. Inference

There are two ways to run inference:

### Method 1: Using `main.py` (Recommended for faster inference)
- Set `mode` to `"predict"` in the `CONFIG` dictionary in `main.py` (default is `"train"`)
- Run inference:
```sh
python main.py
```
- Logs and results are written under the configured `log_dir` (default: `"./logs/<timestamp>/"`)

- (Optional) Configure the following parameters in `CONFIG`:
  - `dataset_path`: Path to your input dataset (default: `"./dataset/data_eval/"`)
  - `para_path`: Path to your trained model parameters (default: `"net_param"`)
  - `base_model_path`: Path to the base model folder (default: `"./base_model"`)
  - `worker_num`: Number of parallel workers for faster inference (default: `8`)

### Method 2: Using `inference.py` (Single-threaded)

- Run inference:
```sh
python inference.py
```
- Results are written under `./logs/<timestamp>/`

- (Optional) Update the `model_path` and `base_model_path` variables near the bottom of `inference.py`

Note: Method 1 (`main.py`) supports multi-threaded inference with configurable `worker_num`, which is significantly faster on large datasets. Method 2 (`inference.py`) is single-threaded and suitable for small datasets.

## Configuration tips

- main.py `CONFIG`:
  - `worker_num`: number of parallel workers for environment simulation. Increase on systems with more CPU cores and sufficient memory.
  - `lr_max`, `lr_min`, `batch_size`, etc.: hyperparameters for training.
  - `log_dir` and `para_path`: control where logs and parameters are saved.

- inference.py:
  - `model_path`: path to the saved model weights (change to your trained `.pth` file).
  - `base_model_path`: path to the `base_model` folder used by the environment.

## QwenVL Baseline Inference (Optional)

### Overview

The QwenVL baseline inference uses QwenVL 7B model with LoRA adapter for architectural layout generation. This is an **optional feature** that requires a separate conda environment. The main DRIP inference pipeline works independently without QwenVL dependencies.

### Architecture

The QwenVL baseline is designed to be completely decoupled from the main codebase:

```
Main DRIP Environment              QwenVL Baseline Environment (Optional)
├── Main inference pipeline        ├── generator_VLM.py
├── env_fixed_state_length.py      ├── utils/make_dataset.py
│   └── Core methods               └── Qwen2.5-VL model
└── No QwenVL imports required
```

### Environment Setup

#### Step 1: Create a separate conda environment

```bash
conda create -n qwenvl python=3.10
conda activate qwenvl
```

#### Step 2: Install QwenVL dependencies

```bash
pip install -r requirements_qwen.txt
```

The `requirements_qwen.txt` file includes:
- All main DRIP_NCv0 dependencies
- QwenVL-specific dependencies:
  - `transformers` - For Qwen2.5-VL model loading
  - `peft` - For LoRA adapter support
  - `qwen-vl-utils` - For vision processing utilities
  - `tqdm` - Progress bars
  - `Pillow` - For image processing

#### Step 3: Prepare model files

Place your model files in the following structure:

```
DRIP_NCv0/
└── base_model/
    ├── Qwen2.5-VL-7B-Instruct/
    │   ├── config.json
    │   ├── model-00001-of-00003.safetensors
    │   └── ...
    └── lora_model/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

### Usage

#### Single File Inference

Edit `generator_VLM.py` and uncomment the single file inference section:

```python
# Single file inference
path_in = "./dataset/data6000_graph/city/"  # Dataset path
file = "city_large_A0.xlsx"
infer(base_model_path, model, processor, path_in, file, output_file_path)
```

Then run:

```bash
python generator_VLM.py
```

#### Batch Inference

The default configuration runs batch inference on all Excel files in the specified directory:

```bash
python generator_VLM.py
```

Results will be saved to `./logs/YYYY_MM_DD_HH_MM_SS/`.

### Input/Output Format

#### Input

- **Excel files** containing floor plan data in the dataset directory
- The system automatically generates environment images and prompts

#### Output

- **Excel files** with two sheets:
  - `floor{N}`: Combined environment data and LLM predictions
  - `floor{N}_llm`: Original LLM output only
- **JPEG images**: Visualizations of the generated layouts

### Code Structure

#### generator_VLM.py

Main inference module containing:
- `load_model()`: Load Qwen2.5-VL model with LoRA adapter
- `run_inference()`: Execute model inference and extract JSON output
- `infer()`: Single file inference wrapper
- `main()`: Batch inference entry point

#### utils/make_dataset.py

Prompt generation utilities:
- `single_floor_prompt()`: Generate prompt for single floor
- `second_floor_prompt()`: Generate prompt for second floor with first floor context
- `generate_prompt_from_file()`: Main prompt generation function

#### env_fixed_state_length.py

Extended with optional QwenVL methods:
- `prepare_prompt()`: Generate environment image and prompt messages
- `save_llm_result()`: Save LLM inference results to Excel

### Important Notes

1. **Decoupled Design**: The main DRIP_NCv0 code does not import QwenVL modules. If QwenVL dependencies are not installed, calling `prepare_prompt()` or `save_llm_result()` will raise an `ImportError`.

2. **Chinese Prompts**: The prompt text remains in Chinese as it was used during model fine-tuning. Only code comments are in English.

3. **Optional Feature**: You can use the main DRIP_NCv0 inference pipeline without installing QwenVL dependencies.

4. **GPU Required**: QwenVL inference requires CUDA-enabled GPU.

### Troubleshooting

#### ImportError: QwenVL dependencies are not installed

**Solution**: Activate the qwenvl conda environment and install dependencies:
```bash
conda activate qwenvl
pip install -r requirements_qwen.txt
```

#### CUDA out of memory

**Solution**: Reduce batch size or use a smaller model variant.

#### Model loading errors

**Solution**: Verify the model paths are correct and all model files are present.

### References

- Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- LoRA: https://github.com/huggingface/peft

## License & citation

See LICENSE file for details.

This repository contains code for training and inference, with a minimal dataset to reproduce key results. Full dataset and training pipeline will be released after further IP protection.

## Contact

For issues, please open an issue in this repository with reproducible steps and the relevant log files found under `logs/`.
