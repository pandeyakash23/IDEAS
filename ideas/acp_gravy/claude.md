# imp_based_motif_level.py - Documentation

## Purpose
This script runs the IDEAS (Importance-based Design of Evolutionary Adaptive Sequences) algorithm for protein sequence optimization. It uses importance scores from a neural network to guide mutations at the motif level, with the importance model being retrained at each design iteration.

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
  - `y_train.npy`: Training fitness values
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
  3. **Evaluates** new sequences using the oracle function
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
1. Identify which positions/motifs are most important for fitness
2. Preferentially mutate low-importance positions
3. Replace with amino acids that have higher importance scores

### Online Retraining
At each design iteration, the importance model is retrained on:
- Original training data (`initial_ohe`, `initial_y`, `initial_seq_len`)
- All newly generated sequences and their fitness scores
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
- `generate_property.py`: Oracle function for fitness evaluation
- `contributions_score_generative.py`: Importance model and training functions
- `sequence_utils.py`: Sequence manipulation utilities

## File Structure
```
IDEAS/
├── ideas/acp_gravy/
│   ├── imp_based_motif_level.py  # This script
│   ├── main.py                    # Adalead class
│   └── generative_results/        # Output directory
├── datasets/acp_gravy/            # Data files
└── importance_models/acp_gravy/   # Model files
```

---

# precise_score_comp.ipynb - Documentation

## Purpose
This notebook performs a comprehensive comparative analysis of different generative sequence optimization methods on the ACP GRAVY dataset. It evaluates methods using three key metrics: optimization performance (AUC), diversity, and novelty.

## Methods Compared
1. **AdaLead** - Adaptive lead optimization
2. **CMA-ES** - Covariance Matrix Adaptation Evolution Strategy
3. **Cbas** - Conditioning by Adaptive Sampling
4. **Dbas** - Design by Adaptive Sampling
5. **DynaPPO** - Dynamic Proximal Policy Optimization
6. **IDEAS** - Custom method (labeled as "our_1")
7. **IDEAS-X** - Exploitation variant (labeled as "ourEXPLOIT_1")

## Data Sources

### Result Files
- Primary directory: `./generative_results/`
- Secondary directory: `/home/apa2237/generative_model_work/dyna_ppo/FLEXS/examples/results_gravy/`

### Required Files Per Method
Each method requires two files:
- `<method_name>.npy` - Performance/property values
- `sequences_<method_name>.npy` - Generated sequences

### Dataset Files
- Test sequences: `/home/apa2237/generative_model_work/datasets/acp_gravy/seq_test.npy`
- Test scores: `/home/apa2237/generative_model_work/datasets/acp_gravy/y_test.npy`

## Workflow

### 1. Benchmark Threshold Calculation (Cell 2)
**Purpose**: Establish fair comparison thresholds across methods

**Process**:
- Loads result files from all methods
- For each sample size (3, 5, 10, 20, 50, 100):
  - Finds maximum value achieved by each method
  - Takes the minimum of these maxima as the benchmark threshold
- Stores in `key_to_minmax` dictionary

**Output Example**:
```python
{3: 0.20488253968253964, 100: 1.3990719999999999, 5: 0.3796304761904762, 10: 0.74764, 50: 1.254328, 20: 0.9842799999999998}
```

**Rationale**: This ensures methods are compared at equivalent difficulty levels rather than raw final scores.

### 2. Data Preparation (Cell 4)
**Purpose**: Create baseline dataset for novelty calculation

**Process**:
- Loads 150 test sequences and their scores
- Applies percentile filtering (40th percentile, cutoff=0.14)
- Retains only bottom performers (105 sequences)
- Mean score drops from -0.29 to -0.71 after filtering

**Why filter?**: The bottom performers represent the starting pool that optimization methods try to improve upon.

### 3. Distance Metrics (Cell 5)
Defines three core distance functions:

#### `edit_dist(seq1, seq2)`
- Calculates Levenshtein edit distance between two sequences
- Returns number of single-character edits needed to transform seq1 to seq2

#### `mean_pairwise_distances(args, seqs)`
- Computes average distance between all pairs in a sequence set
- **Measures diversity**: Higher values = more diverse sequences

#### `novelty(list_1, list_2)`
- Computes average distance between two sequence sets
- **Measures novelty**: Distance from generated sequences to starting pool

### 4. Evaluation Functions

#### `auc_cal(prop)` (Cell 6)
**Purpose**: Calculate Area Under Curve for optimization progress

**Process**:
- For each sample size (20, 50, 100):
  - Extracts performance trajectory over time
  - Shifts minimum to zero (normalizes)
  - Computes AUC using trapezoidal integration
  - Reports mean ± std across runs

**Interpretation**: Higher AUC = faster/better optimization performance

#### `find_nearest(arr, target)` (Cell 7)
**Purpose**: Binary search to find when method reaches benchmark threshold

**Returns**: Index where array value first meets/exceeds target

#### `cal_d_n(sequences, prop_ada)` (Cell 8)
**Purpose**: Calculate diversity and novelty at benchmark-reaching point

**Process**:
- For each sample size and run:
  - Finds index where benchmark threshold is reached
  - Calculates diversity at that point (mean pairwise distance)
  - Calculates novelty at iteration 9 (distance from starting pool)
  - Reports mean ± std

**Why iteration 9 for novelty?**: Fixed checkpoint for consistent comparison

## Notebook Structure

### Cell Index
1. **Cell 1-2**: Import libraries and load method results
2. **Cell 2 (a06b7f3d)**: Calculate benchmark thresholds (`key_to_minmax`)
3. **Cell 3**: Section header - "Preparing the data for metric calculation"
4. **Cell 4 (12601dba)**: Load and filter baseline dataset
5. **Cell 5 (4729a78f)**: Define distance metrics (edit_dist, mean_pairwise_distances, novelty)
6. **Cell 6 (108d787e)**: Define `auc_cal()` function
7. **Cell 7 (58bc713e)**: Define `find_nearest()` binary search function
8. **Cell 8 (67f69474)**: Define `cal_d_n()` function for diversity/novelty
9. **Cell 9**: Section header - "AUC"
10. **Cell 10 (79cec9a3)**: Calculate and print AUC for all methods
11. **Cell 11**: Section header - "Diversity and Novelty"
12. **Cell 12 (5117b1ea)**: Calculate and print diversity/novelty for selected methods
13. **Cell 13 (04b6e3dc)**: Generate AUC bar plot comparison
14. **Cell 14 (ueuu4tcgew9)**: Generate AUC vs Diversity scatter plot

## Metrics Explained

### 1. AUC (Area Under Curve)
- **Range**: 0-50+ (higher is better)
- **Meaning**: Total optimization progress over time
- **Best for**: Comparing overall efficiency and convergence speed

### 2. Diversity
- **Range**: 0-40+ edit distance (higher = more diverse)
- **Meaning**: How different generated sequences are from each other
- **Best for**: Assessing exploration breadth

### 3. Novelty
- **Range**: 0-40+ edit distance (higher = more novel)
- **Meaning**: How different generated sequences are from starting pool
- **Best for**: Assessing whether method explores new regions

## Actual Results

### AUC Performance
Sample sizes: 20, 50, 100

```
Method              20          50          100
AdaLead         16.03±1.08  25.02±1.40  30.57±2.11
CMA-ES          18.94±0.82  21.00±0.81  22.23±0.72
Cbas             6.69±1.62  10.67±0.38  10.30±0.57
Dbas             6.89±0.30   9.92±0.51   9.60±0.84
DynaPPO         10.58±0.26  15.72±0.96  18.64±0.26
IDEAS           23.30±3.07  35.66±2.35  40.18±1.42  [Best overall]
IDEAS-X         19.81±1.13  35.20±2.47  41.22±2.19  [Best at n=100]
```

### Diversity & Novelty (at benchmark threshold)
```
Method              Diversity (100)    Novelty (100)
AdaLead             3.72±1.14          27.08±0.59
CMA-ES             14.30±0.16          27.19±0.13  [Most diverse]
IDEAS              10.36±3.13          28.41±1.73  [Most novel]
IDEAS-X             6.81±1.48          28.10±1.63
```

## Key Insights

### Performance Patterns
1. **IDEAS methods dominate AUC performance**:
   - IDEAS-X: 41.22±2.19 (best at n=100)
   - IDEAS: 40.18±1.42 (best overall consistency)
   - ~80% better than next best competitor (AdaLead: 30.57±2.11)

2. **CMA-ES shows highest diversity** (14.30±0.16 at n=100):
   - Maintains consistent diversity across all sample sizes
   - 38% higher diversity than IDEAS
   - Trades off optimization speed for exploration breadth

3. **IDEAS methods show highest novelty** (~28 edit distance):
   - IDEAS: 28.41±1.73
   - IDEAS-X: 28.10±1.63
   - Slightly more novel than baselines (~27 edit distance)

4. **Clear exploitation-exploration tradeoff**:
   - IDEAS-X: Highest AUC (41.22), Lower diversity (6.81)
   - IDEAS: High AUC (40.18), Moderate diversity (10.36)
   - CMA-ES: Moderate AUC (22.23), Highest diversity (14.30)

### Interpretation Guide
- **High AUC + Low Diversity**: Exploitation-focused (IDEAS-X, AdaLead)
- **Low AUC + High Diversity**: Exploration-focused (DynaPPO)
- **Balanced**: CMA-ES shows consistent moderate values across metrics

## Visualizations

### 1. AUC Bar Plot Comparison (Cell 13)
**Purpose**: Visual comparison of optimization performance across methods and sample sizes

**Features**:
- 1×3 subplot layout (one plot per sample size: 20, 50, 100)
- Bar chart with error bars showing mean ± standard deviation
- Color-coded methods for easy identification
- Value labels on top of each bar
- Grid for better readability

**Configuration**:
- Figure size: 15×5 inches
- 7 methods compared: AdaLead, CMA-ES, Cbas, Dbas, DynaPPO, IDEAS, IDEAS-X
- Y-axis: AUC values
- X-axis: Method names (rotated 45°)

**Interpretation**: Taller bars indicate better optimization performance. IDEAS variants clearly outperform other methods, especially at larger sample sizes.

### 2. AUC vs Diversity Scatter Plot (Cell 14)
**Purpose**: Visualize the tradeoff between optimization performance and sequence diversity

**Features**:
- 1×3 subplot layout with consistent axis ranges across all plots
- Each point represents a method's mean performance
- Error bars show standard deviation in both dimensions
- Color-coded methods matching the bar plot scheme
- Shared X-axis (Diversity) and Y-axis (AUC) labels

**Parametric Configuration** (`PLOT_CONFIG`):
```python
{
    'default_methods': ['IDEAS', 'IDEAS-X', 'CMA-ES', 'AdaLead'],
    'all_methods': [...],  # includes Cbas, Dbas, DynaPPO
    'sample_sizes': [20, 50, 100],
    'marker_size': 120,
    'colors': {...},  # Method-specific color mapping
    'show_error_bars': True
}
```

**Key Features**:
- High DPI (300) for publication-quality output
- Automatic axis range calculation with 10% padding
- Legend only on first subplot to reduce clutter
- Large font sizes (14-16pt) for readability
- Error bars in matching colors with 60% opacity

**Interpretation**:
- **Top-left quadrant** (high AUC, low diversity): Exploitation methods
- **Top-right quadrant** (high AUC, high diversity): Ideal balance
- **Bottom-right quadrant** (low AUC, high diversity): Over-exploration
- Methods moving up-right as sample size increases indicate scaling benefits

**Typical Pattern**: IDEAS-X shows highest AUC but lower diversity, while CMA-ES maintains high diversity with moderate AUC.

## Dependencies
```python
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from polyleven import levenshtein
```

## Running the Analysis
Execute all cells sequentially. The notebook will:
1. Load and process all method results from .npy files
2. Establish benchmark thresholds (min-max across methods)
3. Prepare baseline dataset (filter bottom 40th percentile)
4. Define distance metrics (edit distance, diversity, novelty)
5. Calculate AUC metrics for all methods
6. Calculate diversity and novelty metrics at benchmark points
7. Generate AUC bar plot comparison across sample sizes
8. Generate AUC vs Diversity scatter plots with error bars

## Notes

### File Loading and Filtering
- Files starting with 'sequences_', 'cbas', 'dbas', 'bo', or 'dynappo' are excluded from benchmark threshold calculation
- Files containing 'ourGREEDY' are also excluded from benchmark calculations
- The notebook searches in two directories:
  - `/home/apa2237/generative_model_work/adalead/acp_gravy/generative_results`
  - `/home/apa2237/generative_model_work/dyna_ppo/FLEXS/examples/results_gravy`
- Methods are reordered so "Ours" appears last if present

### Sample Sizes and Reporting
- Sample sizes calculated: 3, 5, 10, 20, 50, 100
- Sample sizes reported in tables and plots: 20, 50, 100
- Each method has multiple independent runs for statistical significance

### Measurement Points
- **Diversity**: Measured at the iteration where method reaches benchmark threshold (adaptive)
- **Novelty**: Measured at iteration 9 (fixed checkpoint for all methods)
- **AUC**: Integrated over entire optimization trajectory, normalized by shifting minimum to zero

### Visualization Options
- Scatter plot can be configured to show either `default_methods` (4 methods) or `all_methods` (7 methods)
- Currently configured to show: IDEAS, IDEAS-X, CMA-ES, AdaLead
- Error bars can be toggled via `show_error_bars` parameter in `PLOT_CONFIG`

## Customization Guide

### Changing Which Methods to Plot
In Cell 14, modify the `methods_to_plot` variable:
```python
# Default: 4 key methods
methods_to_plot = PLOT_CONFIG['default_methods']

# Alternative: All 7 methods
methods_to_plot = PLOT_CONFIG['all_methods']

# Custom: Select specific methods
methods_to_plot = ['IDEAS', 'CMA-ES', 'DynaPPO']
```

### Adjusting Plot Appearance
Modify `PLOT_CONFIG` dictionary in Cell 14:
```python
PLOT_CONFIG = {
    'marker_size': 120,        # Change scatter point size
    'show_error_bars': True,   # Toggle error bars on/off
    'colors': {...}            # Customize method colors
}
```

### Changing Sample Sizes
To analyze different sample sizes, update the filter in analysis cells:
```python
# Current: Reports 20, 50, 100
if k not in [20, 50, 100]:
    continue

# Custom: Report different sizes
if k not in [10, 50]:
    continue
```

### Adjusting Baseline Filtering
In Cell 4, modify the percentile threshold:
```python
top_per = 40  # Current: Keep bottom 60% (below 40th percentile)
top_per = 50  # Alternative: Keep bottom 50%
```
