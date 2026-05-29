# IDEAS
Interpretability Driven Evolutionary Approach for the Design of Biological Sequences

## Overview
IDEAS is a sequence optimization framework that uses importance scores from neural networks to guide evolutionary mutations. The importance model is retrained at each design iteration on all accumulated data, allowing it to adapt as the sequence distribution evolves.

## Installation

Create the conda environment:
```bash
conda env create -f environment.yml
conda activate ideas
```

### Requirements
- Python 3.10
- PyTorch >= 2.0 (with CUDA 11.8)
- NumPy
- pandas
- scikit-learn
- matplotlib
- tensorboard
- Jupyter
- polyleven (for edit distance calculations)

## Project Structure
```
IDEAS/
├── datasets/
│   └── acp_gravy/
│       ├── generate_property.py      # Oracle function & data generation
├── importance_models/
│   └── acp_gravy/
│       ├── train_model.py            # Train importance model
│       ├── contributions_score_generative.py  # Importance scoring
│       ├── complor.py                # Model architecture
│       └── model/                    # Saved model weights
└── ideas/
    └── acp_gravy/
        ├── imp_based_motif_level.py  # Main optimization script
        ├── main.py                   # Adalead algorithm
        ├── precise_score_comp.ipynb  # Results analysis
        └── generative_results/       # Output directory
```

## Usage

### Step 1: Generate/Prepare Data
Ensure your dataset files are in `datasets/acp_gravy/`:
- `x_train.npy`, `y_train.npy`, `len_train.npy` - Training data (one-hot encoded)
- `x_test.npy`, `y_test.npy`, `len_test.npy` - Test data
- `seq_test.npy` - Test sequences (string format)
- `categorical_variables.npy` - Amino acid alphabet

To regenerate the categorical variables:
```bash
cd datasets/acp_gravy
python generate_property.py
```

### Step 2: Train Importance Model (Optional)
If you need to retrain the importance model from scratch:
```bash
cd importance_models/acp_gravy
python train_model.py --num_epochs 2500 --device cuda
```

### Step 3: Run Sequence Optimization
Run the IDEAS optimization algorithm:
```bash
cd ideas/acp_gravy
python imp_based_motif_level.py --temp 1.0 --device cuda
```

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temp` | float | 1.0 | Temperature for softmax mutation selection. Lower values = more exploitation, higher = more exploration |
| `--device` | str | 'cuda' | Device for computation ('cuda' or 'cpu') |

**What it does:**
- Loads test sequences and filters bottom 40th percentile as starting pool
- Runs 10 independent trials with 10 design iterations each
- At each iteration: retrains importance model (50 epochs) on all data, then proposes new sequences
- Tests sample sizes: 20, 50, 100 sequences per batch
- Saves results to `generative_results/`

**Output files:**
- `our_{temp}.npy` - Performance metrics (mean, max fitness per iteration)
- `sequences_our_{temp}.npy` - Generated sequences
- `our_iteration_times_{temp}.npy` - Timing data

### Step 4: Analyze Results
Open and run the Jupyter notebook for comparative analysis:
```bash
cd ideas/acp_gravy
jupyter notebook precise_score_comp.ipynb
```

This notebook calculates:
- **AUC**: Area under the optimization curve (higher = better)
- **Diversity**: Mean pairwise edit distance between generated sequences
- **Novelty**: Edit distance from generated sequences to starting pool

## Key Parameters

### In `imp_based_motif_level.py`:
```python
top_per = 40          # Percentile for filtering starting sequences
num_trials = 10       # Number of independent runs
rounds = 10           # Design iterations per trial
all_ng = [20,50,100]  # Batch sizes to test
retrain_epochs = 50   # Epochs for importance model retraining
motif_size = 1        # Motif size for mutations
```

### In `main.py` (Adalead class):
```python
threshold = 0.05      # Fitness threshold for parent selection
recomb_rate = 0.2     # Recombination rate
mu = 1                # Expected mutations per sequence
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{
anonymous2026textttideas,
title={\${\textbackslash}texttt\{{IDEAS}\}\$: Interpretability Driven Evolutionary Approach for the Design of Biological Sequences},
author={Akash Pandey and Wei Chen and Sinan Keten},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=gaq60U4jvU}
}
```
