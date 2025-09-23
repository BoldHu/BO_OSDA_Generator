# Contras-STE: Contrastive Learning-based SMILES Transformer Encoder for Molecrlar Representation

**Paper Title:**
**Enhancing Conditional Molecular Generation with Pretrained SMILES Transformer and Contrastive Representation Learning**

## Overview
Contras-STE is a research codebase that combines a pretrained SMILES Transformer encoder with contrastive representation learning to improve conditional molecular generation for templating zeolites. 
The workflow first extracts structural descriptors for zeolite frameworks and categorical features for synthesis conditions, and then aligns those conditioning signals with templated organic
directing agent (OSDA) SMILES strings. A GPT-style decoder consumes the aligned representations to
produce SMILES sequences that satisfy user-defined zeolite and synthesis targets.

Key capabilities include:

- **Feature engineering for zeolite synthesis data** – utilities convert raw literature tables into
  numerical descriptors for zeolite topology and synthesis additives that drive the conditional
  generator.【F:utils/data_processing.py†L1-L81】
- **Token-level data augmentation** – the data loader can randomize SMILES strings, build vocabularies,
  and create positive/negative views to support contrastive training of the encoder.【F:datasets/data_loader.py†L14-L166】【F:utils/build_vocab.py†L1-L118】
- **Contrastive SMILES encoder** – a transformer encoder is trained with an InfoNCE objective to align
  anchors, augmented positives, and hard negatives before decoding.【F:models/trfm.py†L1-L67】【F:models/loss.py†L31-L93】
- **Conditional GPT generator** – the autoregressive decoder combines token, property, and positional
  embeddings so that zeolite and synthesis descriptors steer molecular generation.【F:models/GPT.py†L12-L116】
- **Domain-specific evaluation metrics** – scripts compute validity, uniqueness, novelty, FCD, internal
  diversity, and KL divergence for generated SMILES libraries.【F:utils/metrics.py†L1-L126】

The repository accompanies the paper *"Enhancing Conditional Molecular Generation with Pretrained
SMILES Transformer and Contrastive Representation Learning"* and provides end-to-end resources for
reproducing the experiments in zeolite OSDA design.

## Repository Structure

**Paper Title:**
**Enhancing Conditional Molecular Generation with Pretrained SMILES Transformer and Contrastive Representation Learning**
```
.
├── configs/                  # Hyperparameter configuration modules
├── data/                     # Curated zeolite/OSDA spreadsheets and intermediate files
├── data_AFI/                 # Example AFI zeolite conditioning data (SMILES, synthesis, zeolite features)
├── datasets/                 # PyTorch dataset definitions and preprocessing notebooks
├── experiments/              # Training notebooks for the baseline conditional generator
├── experiments_contrastive/  # Contrastive training notebook for Contras-STE
├── generation/               # Sample generations (e.g., AFI_generate_smiles.csv)
├── models/                   # GPT decoder, transformer encoder, and loss definitions
├── utils/                    # Data featurization, vocabulary building, metrics, and helpers
└── figures/                  # Pipeline visualizations used in the manuscript
```

line exhibit average F1-score improvements of 3.12, 0.64, and 1.53 across different tasks, demonstrating its substantial potential for NLP tasks within the zeolite research community.
## Installation

## Pipeline Architecture
The codebase was tested with Python 3.8+ and PyTorch ≥1.5. Install the dependencies using pip:

![Pipeline Overview](./figures/pipeline.png)
```bash
pip install -r requirements.txt
```

## Installation
The requirements file lists exact versions for the core libraries (PyTorch, Hugging Face Transformers,
RDKit, NumPy, tqdm, and seqeval).【F:requirements.txt†L1-L6】

### Requirements
## Data Preparation

* PyTorch (≥1.5.0)
* Transformers (tested on v3.0.2)
* tqdm (≥4.36.0)
* numpy (≥1.18.0)
* seqeval
* rdkit
1. **Collect zeolite/OSDA records** – the `data/OSDA_ZEO.xlsx` spreadsheet stores the curated literature
   dataset. Additional conditioning data for the AFI framework are provided under `data_AFI/`.
2. **Featurize zeolite and synthesis conditions** – use the helper functions in
   `utils/data_processing.py` to convert rows into zeolite descriptors and synthesis indicator
   vectors.【F:utils/data_processing.py†L1-L81】
3. **Augment and split SMILES** – `datasets/data_loader.py` exposes dataset classes that randomize
   SMILES strings, pad them to fixed lengths, and split into training and evaluation subsets. The
   `Randomizer` class can generate randomized SMILES on the fly, while `data_split` provides random or
   zeolite-based partitions.【F:datasets/data_loader.py†L38-L146】【F:utils/utils.py†L49-L103】
4. **Build the tokenizer** – generate a vocabulary from a SMILES corpus with
   `utils/build_vocab.py`. The resulting `vocab.pkl` is stored under `model_hub/` and reused by the data
   loaders.【F:utils/build_vocab.py†L1-L147】

Install dependencies via:
Running the preprocessing notebooks in `datasets/data_preprocessing.ipynb` or the scripts in `utils/`
will create the NumPy arrays and CSV files consumed by the training notebooks.

```bash
pip install torch transformers tqdm numpy seqeval rdkit
```
## Training Workflow

1. **Pretrain the SMILES encoder** – launch `experiments/train_GPT_vocab.ipynb` to pretrain the
   transformer encoder on the OSDA corpus. The notebook builds datasets with `Seq2seqDataset` and trains
   the `TrfmSeq2seq` model defined in `models/trfm.py` to reconstruct SMILES tokens.【F:datasets/data_loader.py†L85-L124】【F:models/trfm.py†L1-L57】
2. **Contrastive fine-tuning** – open `experiments_contrastive/train_GPT_contrastive.ipynb` to minimize
   the InfoNCE objective implemented in `models/loss.py`. `Contrastive_Seq2seqDataset` samples positive
   variants from canonical SMILES clusters and negatives from a large pool (e.g., ChEMBL) to sharpen the
   encoder representations.【F:datasets/data_loader.py†L126-L200】【F:models/loss.py†L31-L93】
3. **Conditional GPT training** – the conditional generator in `models/GPT.py` receives the frozen (or
   jointly trained) encoder embeddings alongside zeolite/synthesis descriptors. The GPTConfig block size,
   embedding dimensions, dropout, and number of transformer layers can be adjusted through the
   configuration module in `configs/config_clamer.py`.【F:models/GPT.py†L12-L116】【F:configs/config_clamer.py†L1-L12】

Each notebook saves checkpoints and tokenizer assets into `model_hub/`. Feel free to convert the
notebooks into Python scripts for large-scale experiments.

## Molecular Generation

After training, condition the GPT decoder on desired zeolite and synthesis descriptors to produce
SMILES tokens autoregressively. The helper routine `predict` in `utils/utils.py` demonstrates how to
seed the decoder with start tokens and iteratively append the highest-probability character at each
step.【F:utils/utils.py†L105-L156】 Generated molecules for the AFI case study are stored in
`generation/AFI_generate_smiles.csv` for reference.

## Data and Pre-trained Models
## Evaluation

### Datasets
Quantitatively assess generated libraries with the metrics implemented in `utils/metrics.py`. The
module reports validity, uniqueness, reconstructability, novelty, Fréchet ChemNet Distance (FCD),
internal diversity, and KL divergence between molecular fingerprint distributions.【F:utils/metrics.py†L1-L126】
The `calculate_KL.ipynb` notebook gives an example of computing the KL divergence over Morgan
fingerprints.

## License

* [Pre-trained Model and Fine-tuned Models](https://www.kaggle.com/datasets/boldhu/original-dataset-for-zeolbert)
This project is released for academic research. Please contact the authors for commercial licensing
inquiries.
