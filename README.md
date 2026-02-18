# NAS Optimization for Multimodal VLM Skin Lesion Recognition

This repository was developed for the paper:
`Neural Architecture Search optimization for a multimodal model for skin lesion recognition - Case study`.

## Run VLM Multimodal Experiments

### 1) Configure environment variables

Go to the project root and create your local `.env` file:

```bash
cp .env.example .env
```

Then edit `.env` with your experiment setup. Example:

```env
NUM_EPOCHS=100
BATCH_SIZE=8
K_FOLDS=5
LIST_NUM_HEADS=[8]
COMMON_DIM=512
DATASET_FOLDER_NAME="PAD-UFES-20"
DATASET_FOLDER_PATH="./data/PAD-UFES-20"
save_to_disk=True
RESULTS_FOLDER_PATH="./data/PAD-UFES-20/results/artigo1_13022026/patient-level"
```

#### Environment variables explained

- `NUM_EPOCHS`: Number of training epochs per fold/model. Higher values usually improve convergence, but increase runtime.
- `BATCH_SIZE`: Number of samples per gradient update. Larger batches use more GPU memory.
- `K_FOLDS`: Number of folds used in cross-validation.
- `LIST_NUM_HEADS`: Number of attention heads in multimodal fusion modules.
- `COMMON_DIM`: Size of the shared latent projection space used to align modalities.
- `DATASET_FOLDER_NAME`: Dataset folder identifier (used in naming/organization).
- `DATASET_FOLDER_PATH`: Full path to the dataset root directory used by training scripts.
- `save_to_disk`: If `True`, saves trained model weights and artifacts.
- `RESULTS_FOLDER_PATH`: Output path where metrics, logs, and saved artifacts are written.

### 2) Define the NAS search space

Set the search space in:
`src/nas/optimization_train_process_pad_20_using_llm_as_controller.py`

```python
search_space = {
    "num_blocks": [2, 5, 10],
    "initial_filters": [16, 32, 64],
    "kernel_size": [3, 5],
    "layers_per_block": [1, 2],
    "use_pooling": [True, False],
    "common_dim": [64, 128, 256, 512],
    "attention_mecanism": ["concatenation", "crossattention", "metablock", "gfcam"],
    "num_layers_text_fc": [1, 2, 3],
    "neurons_per_layer_size_of_text_fc": [64, 128, 256, 512],
    "num_layers_fc_module": [1, 2],
    "neurons_per_layer_size_of_fc_module": [256, 512],
}
```

### 3) Define NAS search steps

In the same file, set:

```python
SEARCH_STEPS = 500
```

Increase or reduce this value based on compute budget and desired exploration depth.

### 4) Run multimodal VLM training/optimization

Option A (Python script):

```bash
python3 ./src/nas/optimization_train_process_pad_20_using_llm_as_controller.py
```

Option B (Bash runner):

```bash
./src/nas/run_script_via_bash.sh
```
