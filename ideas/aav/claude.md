# imp_based_motif_level.py - Documentation (AAV Fitness)

## Purpose
This script runs the IDEAS (Importance-based Design of Evolutionary Adaptive Sequences) algorithm for AAV (Adeno-Associated Virus) capsid protein sequence optimization. It uses importance scores from a neural network to guide mutations at the motif level, with the importance model being retrained at each design iteration.

## AAV Fitness
This dataset targets AAV capsid protein fitness optimization. The fitness is predicted using a neural network model (same model used for importance scores). Higher values indicate better fitness/viability of the AAV capsid variant.

## Data Source
The AAV dataset is from the FLIP benchmark: https://flip.protein.properties/

## Command Line Arguments
```bash
python imp_based_motif_level.py --temp 1.0 --device cuda
```
- `--temp`: Temperature parameter for softmax-based mutation selection (default: 1.0)
- `--device`: CUDA device for computation (default: 'cuda')

## Workflow

### 1. Data Loading and Filtering
**Lines 36-60**

- Loads test sequences (`seq_test.npy`) and their fitness scores (`y_test.npy`)
- Applies cutoff filtering to keep only sequences with y < 0.2
- Sorts sequences by fitness (descending order)
- These filtered sequences serve as the starting pool for optimization

### 2. Model Initialization
**Lines 70-97**

- Loads the pre-trained importance/oracle model (`model_contri`) using `initalize_contri()`
- This model serves dual purpose: predicting fitness AND computing importance scores
- Loads initial training data for importance model retraining:
  - `x_train.npy`: One-hot encoded training sequences
  - `y_train.npy`: Training fitness values
  - `len_train.npy`: Sequence lengths

### 3. Adalead Algorithm Setup
**Lines 116-137**

Creates an `Adalead` instance with:
- `motif_based=True`: Uses motif-level mutations
- `motif_size=15`: 15 amino acid motifs (larger due to longer sequences)
- `is_imp_based=True`: Uses importance-guided mutations
- `retrain_epochs=50`: Retrains importance model for 50 epochs each iteration
- Initial training data passed for online retraining

### 4. Experiment Loop
**Lines 148-178**

For each sample size (`ng` in [20, 50, 100]):
- Runs `num_trials=10` independent trials
- Each trial runs `rounds=10` design iterations
- In each iteration:
  1. **Retrains** the importance model on all data (original + generated)
  2. **Proposes** new sequences using importance-guided mutations
  3. **Evaluates** new sequences using the neural network oracle
  4. **Tracks** mean, max fitness, and timing

### 5. Results Saving
**Lines 180-184**

Saves three files to `generative_results/`:
- `our_{temp}.npy`: Performance metrics (mean, max, new_mean) per trial/round
- `sequences_our_{temp}.npy`: Generated sequences per trial/round
- `our_iteration_times_{temp}.npy`: Time taken per iteration

## Key Components

### Neural Network Oracle
Unlike other datasets (ACP, TF1) that use formula-based or lookup-based oracles, AAV uses the same neural network model for:
1. **Fitness prediction** (`predict_property` function)
2. **Importance score computation** (`contribution_score` function)

### Importance-Based Mutation
The algorithm uses importance scores to:
1. Identify which positions/motifs are most important for fitness
2. Preferentially mutate low-importance positions
3. Replace with amino acids that have higher importance scores

### Online Retraining
At each design iteration, the importance model is retrained on:
- Original training data (`initial_ohe`, `initial_y`, `initial_seq_len`)
- All newly generated sequences and their predicted fitness scores
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
- `contributions_score_generative.py`: Neural network model, importance scores, and training functions
- `sequence_utils.py`: Sequence manipulation utilities

## File Structure
```
IDEAS/
├── ideas/aav/
│   ├── imp_based_motif_level.py  # This script
│   ├── main.py                    # Adalead class
│   └── generative_results/        # Output directory
├── datasets/aav/                  # Data files
└── importance_models/aav/         # Model files
```

## Differences from Other Datasets
- **Sequence Type**: Protein (20 amino acids)
- **Sequence Length**: Up to 750 amino acids (much longer than ACP/TF1)
- **Motif Size**: 15 (larger to account for longer sequences)
- **Oracle**: Neural network-based (same model as importance scorer)
- **Property**: AAV capsid fitness/viability
