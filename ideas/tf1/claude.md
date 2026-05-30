# imp_based_motif_level.py - Documentation (TF1 Binding)

## Purpose
This script runs the IDEAS (Importance-based Design of Evolutionary Adaptive Sequences) algorithm for DNA sequence optimization targeting **Transcription Factor 1 (TF1) binding affinity**. It uses importance scores from a neural network to guide mutations at the motif level, with the importance model being retrained at each design iteration.

## TF1 Binding
This dataset targets the ARX transcription factor (L343Q variant) binding to 8-mer DNA sequences. The binding affinity is measured experimentally and stored in a lookup dictionary. Higher values indicate stronger binding.

## Command Line Arguments
```bash
python imp_based_motif_level.py --temp 1.0 --device cuda
```
- `--temp`: Temperature parameter for softmax-based mutation selection (default: 1.0)
- `--device`: CUDA device for computation (default: 'cuda')

## Workflow

### 1. Data Loading and Filtering
**Lines 39-63**

- Loads test sequences (`seq_test.npy`) and their fitness scores (`y_test.npy`)
- Applies cutoff filtering to keep only sequences with y < 0.2
- Sorts sequences by fitness (descending order)
- These filtered sequences serve as the starting pool for optimization

### 2. Model Initialization
**Lines 79-86**

- Loads the pre-trained importance model (`model_contri`) using `initalize_contri()`
- Loads initial training data for importance model retraining:
  - `x_train.npy`: One-hot encoded training sequences
  - `y_train.npy`: Training fitness values (binding affinity)
  - `len_train.npy`: Sequence lengths

### 3. Adalead Algorithm Setup
**Lines 106-127**

Creates an `Adalead` instance with:
- `motif_based=True`: Uses motif-level mutations
- `motif_size=1`: Single nucleotide motifs
- `is_imp_based=True`: Uses importance-guided mutations
- `retrain_epochs=50`: Retrains importance model for 50 epochs each iteration
- Initial training data passed for online retraining

### 4. Experiment Loop
**Lines 138-171**

For each sample size (`ng` in [20, 50, 100]):
- Runs `num_trials=10` independent trials
- Each trial runs `rounds=10` design iterations
- In each iteration:
  1. **Retrains** the importance model on all data (original + generated)
  2. **Proposes** new sequences using importance-guided mutations
  3. **Evaluates** new sequences using the binding affinity oracle function
  4. **Tracks** mean, max fitness, and timing

### 5. Results Saving
**Lines 174-178**

Saves three files to `generative_results/`:
- `our_{temp}.npy`: Performance metrics (mean, max, new_mean) per trial/round
- `sequences_our_{temp}.npy`: Generated sequences per trial/round
- `our_iteration_times_{temp}.npy`: Time taken per iteration

## Key Components

### Importance-Based Mutation
The algorithm uses importance scores to:
1. Identify which positions/motifs are most important for binding affinity
2. Preferentially mutate low-importance positions
3. Replace with nucleotides that have higher importance scores

### Online Retraining
At each design iteration, the importance model is retrained on:
- Original training data (`initial_ohe`, `initial_y`, `initial_seq_len`)
- All newly generated sequences and their binding affinity scores
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
- `generate_property.py`: Oracle function for binding affinity lookup
- `contributions_score_generative.py`: Importance model and training functions
- `sequence_utils.py`: Sequence manipulation utilities

## File Structure
```
IDEAS/
├── ideas/tf1/
│   ├── imp_based_motif_level.py  # This script
│   ├── main.py                    # Adalead class
│   └── generative_results/        # Output directory
├── datasets/tf1/                  # Data files
└── importance_models/tf1/         # Model files
```

## Differences from ACP datasets
- **Sequence Type**: DNA (A, G, C, T) instead of protein (20 amino acids)
- **Sequence Length**: Fixed 8-mer sequences
- **Property**: Transcription factor binding affinity
- **Oracle**: Lookup-based (pre-computed binding values) instead of formula-based
