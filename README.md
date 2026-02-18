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

# Paper production: Neural Architecture Search optimization for a multimodal model for skin lesion recognition - Case study

This repository is part of a Neural Architecture Search study to optimize a VLM model for skin lesions classification when using the PAD-UFES-20 dataset.

Please refer to the paper when using this repository. Paper "Neural Architecture Search optimization for a multimodal model for skin lesion recognition - Case study."

## Supplementary material

This Supplementary material section reports the configurations of the top-10 architectures identified by the NAS process, ranked in descending order according to the A-TOPSIS multi-criteria decision-making score. These results provide additional transparency regarding the architectural trade-offs explored during search and support the selection of the final model (ID~21).

\begin{table*}[!ht]
\centering
\caption{Top-10 multimodal architectures discovered by NAS, ranked in descending order according to the A-TOPSIS score. The \emph{History} column specifies how validation metrics from previous trials were summarized and provided to the LLM controller.}
\label{tab:top10_architectures}
\resizebox{1.05\textwidth}{!}{%
\begin{tabular}{c|c|c|c|c|c|c|c|c|c}
\hline
Rank & ID & History & CNN Blocks & Init. Filters & Kernel & Layers/Block & Fusion & Fusion Dim & Classifier MLP \\
\hline
1  & 21 & TOP-10-BACC & 10 & 64 & 3 & 2 & MetaBlock       & 512 & 2 $\times$ 512 \\
2  & 23 & TOP-10-BACC & 5  & 64 & 3 & 1 & MetaBlock       & 256 & 1 $\times$ 512 \\
3  & 11 & LAST-10     & 5  & 32 & 3 & 2 & MetaBlock       & 512 & 1 $\times$ 512 \\
4  & 7  & FULL        & 5  & 64 & 3 & 2 & MetaBlock       & 256 & 1 $\times$ 512 \\
5  & 5  & FULL        & 10 & 64 & 5 & 1 & MetaBlock       & 512 & 2 $\times$ 512 \\
6  & 2  & FULL        & 2  & 32 & 5 & 2 & MetaBlock       & 512 & 2 $\times$ 512 \\
7  & 3  & FULL        & 5  & 64 & 3 & 2 & MetaBlock       & 512 & 2 $\times$ 256 \\
8  & 19 & TOP-10-BACC & 5  & 32 & 3 & 1 & MetaBlock       & 128 & 1 $\times$ 512 \\
9  & 15 & LAST-10     & 5  & 64 & 3 & 2 & Cross-Attention & 512 & 1 $\times$ 256 \\
10 & 18 & TOP-10-BACC & 2  & 16 & 5 & 2 & MetaBlock       & 128 & 2 $\times$ 512 \\
\hline
\end{tabular}%
}
\end{table*}

Across the top-10 ranked architectures, MetaBlock emerges as the dominant fusion mechanism, appearing in nearly all high-performing solutions (Appendix~A). In addition, the fusion dimension of 512 is the most frequent configuration among these models, indicating a consistent preference for higher-dimensional shared representations within the explored search space.
