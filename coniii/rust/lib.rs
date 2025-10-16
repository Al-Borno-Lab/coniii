use pyo3::prelude::*;
use numpy::{PyArray2, IntoPyArray};
use numpy::ndarray::Array2;

// Import our core sampling logic
mod core;
use core::samplers::{IsingCore, Potts3Core, SamplerCore, generate_samples, generate_samples_parallel, calculate_means};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// Ising model sampler - Python wrapper around core logic
#[pyclass]
pub struct Ising {
    core: IsingCore,
    sample: Vec<Vec<i32>>,
}

#[pymethods]
impl Ising {
    /// Create a new Ising sampler
    #[new]
    #[pyo3(signature = (n, multipliers, seed=None))]
    fn new(n: usize, multipliers: Vec<f64>, seed: Option<i64>) -> PyResult<Self> {
        let actual_seed = match seed {
            Some(s) if s >= 0 => Some(s as u64),
            _ => None,
        };
        
        let core = IsingCore::new(n, multipliers, actual_seed);
        
        Ok(Ising {
            core,
            sample: Vec::new(),
        })
    }
    
    /// Calculate energy for Ising model
    fn calc_e(&self, config: Vec<i32>) -> PyResult<f64> {
        Ok(self.core.calc_energy(&config))
    }
    
    /// Generate a sample using Metropolis algorithm
    fn sample_metropolis(&mut self, steps: usize) -> PyResult<f64> {
        // Initialize random configuration if no sample exists
        if self.sample.is_empty() {
            let config = self.core.init_config();
            self.sample.push(config);
        }
        
        let mut config = self.sample.last().unwrap().clone();
        
        for _ in 0..steps {
            self.core.metropolis_step(&mut config);
        }
        
        // Update the sample
        if !self.sample.is_empty() {
            self.sample.pop();
        }
        self.sample.push(config.clone());
        
        Ok(self.core.calc_energy(&config))
    }
    
    /// Generate multiple samples
    fn generate_sample(&mut self, n_samples: usize, burn_in: usize, steps: usize, verbose: bool) -> PyResult<()> {
        self.sample = generate_samples(&mut self.core, n_samples, burn_in, steps, verbose);
        Ok(())
    }

    /// Generate multiple samples in parallel using Rayon
    fn generate_sample_parallel(&mut self, n_samples: usize, burn_in: usize, steps: usize, verbose: bool) -> PyResult<()> {
        self.sample = generate_samples_parallel(&self.core, n_samples, burn_in, steps, verbose);
        Ok(())
    }
    
    /// Get the generated sample as a numpy array
    fn fetch_sample(&self, py: Python) -> PyResult<PyObject> {
        if self.sample.is_empty() {
            let empty_array = PyArray2::<f64>::zeros_bound(py, [0, self.core.system_size()], false);
            return Ok(empty_array.into_py(py));
        }
        
        let n_samples = self.sample.len();
        let n_vars = self.core.system_size();
        let mut array = Array2::zeros((n_samples, n_vars));
        
        for (i, sample) in self.sample.iter().enumerate() {
            for (j, &value) in sample.iter().enumerate() {
                array[[i, j]] = value as f64;
            }
        }
        
        Ok(array.into_pyarray_bound(py).into_py(py))
    }
    
    /// Calculate means of the sample
    fn means(&self) -> PyResult<Vec<f64>> {
        Ok(calculate_means(&self.sample, self.core.system_size()))
    }
    
    /// Print sample information
    fn print(&self, n_print: usize) -> PyResult<()> {
        let n_to_print = n_print.min(self.sample.len());
        println!("Sample (showing first {}):", n_to_print);
        
        for (i, sample) in self.sample.iter().take(n_to_print).enumerate() {
            println!("Sample {}: {:?}", i, sample);
        }
        
        Ok(())
    }
    
    /// Get system size
    #[getter]
    fn n(&self) -> usize {
        self.core.system_size()
    }
    
    /// Get multipliers
    #[getter]
    fn multipliers(&self) -> Vec<f64> {
        self.core.get_multipliers().to_vec()
    }
    
    /// Set multipliers
    #[setter]
    fn set_multipliers(&mut self, multipliers: Vec<f64>) {
        self.core.set_multipliers(multipliers);
    }
    
    /// Set coupling matrix
    fn set_coupling_matrix(&mut self, matrix: Vec<Vec<f64>>) -> PyResult<()> {
        self.core.set_coupling_matrix(matrix);
        Ok(())
    }
    
    /// Get coupling matrix
    fn get_coupling_matrix(&self) -> Vec<Vec<f64>> {
        self.core.get_coupling_matrix().clone()
    }
}

/// 3-state Potts model sampler - Python wrapper around core logic
#[pyclass]
pub struct Potts3 {
    core: Potts3Core,
    sample: Vec<Vec<i32>>,
}

#[pymethods]
impl Potts3 {
    /// Create a new Potts3 sampler
    #[new]
    #[pyo3(signature = (n, multipliers, seed=None))]
    fn new(n: usize, multipliers: Vec<f64>, seed: Option<i64>) -> PyResult<Self> {
        let actual_seed = match seed {
            Some(s) if s >= 0 => Some(s as u64),
            _ => None,
        };
        
        let core = Potts3Core::new(n, multipliers, actual_seed);
        
        Ok(Potts3 {
            core,
            sample: Vec::new(),
        })
    }
    
    /// Calculate energy for Potts model
    fn calc_e(&self, config: Vec<i32>) -> PyResult<f64> {
        Ok(self.core.calc_energy(&config))
    }
    
    /// Generate a sample using Metropolis algorithm
    fn sample_metropolis(&mut self, steps: usize) -> PyResult<f64> {
        // Initialize random configuration if no sample exists
        if self.sample.is_empty() {
            let config = self.core.init_config();
            self.sample.push(config);
        }
        
        let mut config = self.sample.last().unwrap().clone();
        
        for _ in 0..steps {
            self.core.metropolis_step(&mut config);
        }
        
        // Update the sample
        if !self.sample.is_empty() {
            self.sample.pop();
        }
        self.sample.push(config.clone());
        
        Ok(self.core.calc_energy(&config))
    }
    
    /// Generate multiple samples
    fn generate_sample(&mut self, n_samples: usize, burn_in: usize, steps: usize, verbose: bool) -> PyResult<()> {
        self.sample = generate_samples(&mut self.core, n_samples, burn_in, steps, verbose);
        Ok(())
    }

    /// Generate multiple samples in parallel using Rayon
    fn generate_sample_parallel(&mut self, n_samples: usize, burn_in: usize, steps: usize, verbose: bool) -> PyResult<()> {
        self.sample = generate_samples_parallel(&self.core, n_samples, burn_in, steps, verbose);
        Ok(())
    }
    
    /// Get the generated sample as a numpy array
    fn fetch_sample(&self, py: Python) -> PyResult<PyObject> {
        if self.sample.is_empty() {
            let empty_array = PyArray2::<f64>::zeros_bound(py, [0, self.core.system_size()], false);
            return Ok(empty_array.into_py(py));
        }
        
        let n_samples = self.sample.len();
        let n_vars = self.core.system_size();
        let mut array = Array2::zeros((n_samples, n_vars));
        
        for (i, sample) in self.sample.iter().enumerate() {
            for (j, &value) in sample.iter().enumerate() {
                array[[i, j]] = value as f64;
            }
        }
        
        Ok(array.into_pyarray_bound(py).into_py(py))
    }
    
    /// Calculate means of the sample
    fn means(&self) -> PyResult<Vec<f64>> {
        Ok(calculate_means(&self.sample, self.core.system_size()))
    }
    
    /// Print sample information
    fn print(&self, n_print: usize) -> PyResult<()> {
        let n_to_print = n_print.min(self.sample.len());
        println!("Sample (showing first {}):", n_to_print);
        
        for (i, sample) in self.sample.iter().take(n_to_print).enumerate() {
            println!("Sample {}: {:?}", i, sample);
        }
        
        Ok(())
    }
    
    /// Get current value (for state management)
    fn get_value(&self) -> PyResult<Vec<i32>> {
        if self.sample.is_empty() {
            return Ok(vec![0; self.core.system_size()]);
        }
        Ok(self.sample.last().unwrap().clone())
    }
    
    /// Get current state
    fn get_state(&self) -> PyResult<(Vec<i32>, Vec<f64>, Option<u64>)> {
        let config = self.get_value()?;
        Ok((config, self.core.get_multipliers().to_vec(), None))
    }
    
    /// Set state
    fn set_state(&mut self, state: (Vec<i32>, Vec<f64>, Option<u64>)) -> PyResult<()> {
        self.core.set_multipliers(state.1);
        if !state.0.is_empty() {
            self.sample = vec![state.0];
        }
        Ok(())
    }
    
    /// Get system size
    #[getter]
    fn n(&self) -> usize {
        self.core.system_size()
    }
    
    /// Get multipliers
    #[getter]
    fn multipliers(&self) -> Vec<f64> {
        self.core.get_multipliers().to_vec()
    }
    
    /// Set multipliers
    #[setter]
    fn set_multipliers(&mut self, multipliers: Vec<f64>) {
        self.core.set_multipliers(multipliers);
    }
    
    /// Set coupling matrix
    fn set_coupling_matrix(&mut self, matrix: Vec<Vec<f64>>) -> PyResult<()> {
        self.core.set_coupling_matrix(matrix);
        Ok(())
    }
    
    /// Get coupling matrix
    fn get_coupling_matrix(&self) -> Vec<Vec<f64>> {
        self.core.get_coupling_matrix().clone()
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn coniii(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Ising>()?;
    m.add_class::<Potts3>()?;
    Ok(())
}
