# imp_based_motif_level.py - Documentation (Aliphatic Index)

## Purpose
This script runs the IDEAS (Importance-based Design of Evolutionary Adaptive Sequences) algorithm for protein sequence optimization targeting **Aliphatic Index**. It uses importance scores from a neural network to guide mutations at the motif level, with the importance model being retrained at each design iteration.

## Aliphatic Index
Aliphatic index is a measure of the relative volume occupied by aliphatic side chains (Alanine, Valine, Isoleucine, and Leucine). It is calculated as:

```
AI = 100 * (Ala% + 2.9*Val% + 3.9*(Ile% + Leu%))
```

Higher aliphatic index indicates higher thermostability of globular proteins.

## Command Line Arguments
```bash
python imp_based_motif_level.py --temp 1.0 --device cuda
```
- `--temp`: Temperature parameter for softmax-based mutation selection (default: 1.0)
- `--device`: CUDA device for computation (default: 'cuda')

## Workflow

### 1. Data Loading and Filtering
**Lines 38-68**

- Loads test sequences (`seq_test.npy`) and their fitness scores (`y_test.npy`)
- Applies percentile filtering to keep only the bottom performers (40th percentile cutoff)
- Sorts sequences by fitness (descending order)
- These filtered sequences serve as the starting pool for optimization

### 2. Model Initialization
**Lines 87-94**

- Loads the pre-trained importance model (`model_contri`) using `initalize_contri()`
- Loads initial training data for importance model retraining:
  - `x_train.npy`: One-hot encoded training sequences
  - `y_train.npy`: Training fitness values (Aliphatic Index)
  - `len_train.npy`: Sequence lengths

### 3. Adalead Algorithm Setup
**Lines 114-135**

Creates an `Adalead` instance with:
- `motif_based=True`: Uses motif-level mutations
- `motif_size=1`: Single amino acid motifs
- `is_imp_based=True`: Uses importance-guided mutations
- `retrain_epochs=50`: Retrains importance model for 50 epochs each iteration
- Initial training data passed for online retraining

### 4. Experiment Loop
**Lines 146-179**

For each sample size (`ng` in [20, 50, 100]):
- Runs `num_trials=10` independent trials
- Each trial runs `rounds=10` design iterations
- In each iteration:
  1. **Retrains** the importance model on all data (original + generated)
  2. **Proposes** new sequences using importance-guided mutations
  3. **Evaluates** new sequences using the aliphatic index oracle function
  4. **Tracks** mean, max fitness, and timing

### 5. Results Saving
**Lines 182-186**

Saves three files to `generative_results/`:
- `our_{temp}.npy`: Performance metrics (mean, max, new_mean) per trial/round
- `sequences_our_{temp}.npy`: Generated sequences per trial/round
- `our_iteration_times_{temp}.npy`: Time taken per iteration

## Key Components

### Importance-Based Mutation
The algorithm uses importance scores to:
1. Identify which positions/motifs are most important for aliphatic index
2. Preferentially mutate low-importance positions
3. Replace with amino acids that have higher importance scores

### Online Retraining
At each design iteration, the importance model is retrained on:
- Original training data (`initial_ohe`, `initial_y`, `initial_seq_len`)
- All newly generated sequences and their aliphatic index scores
- Training runs for 50 epochs

This allows the model to adapt its importance predictions as the sequence distribution shifts during optimization.

## Output Data Structure

### `our_{temp}.npy`
```python
{
    20: np.array([num_trials, rounds+1, 3]),  # [mean, max, new_mean]
    50: np.array([num_trials, rounds+1, 3]),
    100: np.array([num_trials, rounds+1, 3])
}
```

### `sequences_our_{temp}.npy`
```python
{
    20: {
        trial_0: {round_1: [seq1, seq2, ...], round_2: [...], ...},
        trial_1: {...},
        ...
    },
    50: {...},
    100: {...}
}
```

## Dependencies
- `main.py`: Contains the `Adalead` class
- `generate_property.py`: Oracle function for aliphatic index calculation
- `contributions_score_generative.py`: Importance model and training functions
- `sequence_utils.py`: Sequence manipulation utilities

## File Structure
```
IDEAS/
├── ideas/acp_aliphatic/
│   ├── imp_based_motif_level.py  # This script
│   ├── main.py                    # Adalead class
│   └── generative_results/        # Output directory
├── datasets/acp_aliphatic/        # Data files
└── importance_models/acp_aliphatic/  # Model files
```

## Differences from ACP GRAVY
- **Property**: Aliphatic Index instead of GRAVY (hydropathy)
- **Formula**: AI = 100 * (Ala% + 2.9*Val% + 3.9*(Ile% + Leu%))
- **Coefficients**: Uses aliphatic amino acid coefficients instead of hydropathy values
