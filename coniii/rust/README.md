# Rust Core Implementation

This directory contains the Rust implementation of the statistical physics samplers, separated into core logic and Python bindings.

## File Structure

- **`core.rs`** - Contains all the core sampling algorithms and statistical physics logic
- **`lib.rs`** - Contains Python bindings using PyO3

## Core Logic (`core.rs`)

The core logic is organized into modules:

### `samplers` Module

Contains the main sampling algorithms:

- **`SamplerCore` trait** - Defines the interface for all samplers
- **`IsingCore`** - Ising model implementation
- **`Potts3Core`** - 3-state Potts model implementation
- **Utility functions**:
  - `generate_samples()` - Generic sampling algorithm
  - `calculate_means()` - Calculate sample means
  - `calculate_correlations()` - Calculate pairwise correlations
  - `calculate_energy_stats()` - Energy statistics

### `utils` Module

Statistical analysis utilities:

- `autocorrelation()` - Calculate autocorrelation function
- `integrated_autocorr_time()` - Estimate integrated autocorrelation time
- `effective_sample_size()` - Calculate effective sample size

## Python Bindings (`lib.rs`)

The Python bindings provide a clean interface to the core logic:

- **`Ising`** - Python class wrapping `IsingCore`
- **`Potts3`** - Python class wrapping `Potts3Core`

## Usage

### From Python:
```python
from coniii import Ising, Potts3

# Create samplers
ising = Ising(n=4, multipliers=[0.1, 0.2, 0.3, 0.4], seed=42)
potts = Potts3(n=4, multipliers=[0.1, 0.2, 0.3, 0.4], seed=42)

# Use the samplers
energy = ising.calc_e([1, -1, 1, -1])
ising.generate_sample(n_samples=1000, burn_in=100, steps=10)
sample = ising.fetch_sample()  # Returns numpy array
```

### From Rust (if using as a library):
```rust
use coniii::core::samplers::{IsingCore, SamplerCore};

let mut ising = IsingCore::new(4, vec![0.1, 0.2, 0.3, 0.4], Some(42));
let energy = ising.calc_energy(&[1, -1, 1, -1]);
```

## Benefits of This Structure

1. **Separation of Concerns**: Core algorithms are separate from Python bindings
2. **Testability**: Core logic can be tested independently
3. **Reusability**: Core logic can be used in other contexts (not just Python)
4. **Maintainability**: Easier to modify algorithms without touching Python code
5. **Performance**: Core logic is optimized Rust code

## Adding New Samplers

To add a new sampler:

1. Implement the `SamplerCore` trait in `core.rs`
2. Add a Python wrapper class in `lib.rs`
3. Register the class in the `#[pymodule]` function

## Testing

Run tests with:
```bash
cargo test
```

The core logic includes unit tests for energy calculations and sampling algorithms.
