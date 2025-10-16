// MIT License
// 
// Copyright (c) 2020 Gunnar. E
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use rand::distributions::{Distribution, Uniform};
use std::collections::HashMap;

/// Core sampling engine for statistical physics models
/// This module contains the main algorithms separated from Python bindings
pub mod samplers {
    use super::*;

    /// Base sampler trait that defines the interface for all sampling algorithms
    pub trait SamplerCore {
        /// Calculate the energy of a given configuration
        fn calc_energy(&self, config: &[i32]) -> f64;
        
        /// Perform one step of the Metropolis algorithm
        fn metropolis_step(&mut self, config: &mut Vec<i32>) -> f64;
        
        /// Initialize a random configuration
        fn init_config(&mut self) -> Vec<i32>;
        
        /// Get the system size
        fn system_size(&self) -> usize;
        
        /// Get the current multipliers (fields and couplings)
        fn get_multipliers(&self) -> &[f64];
        
        /// Set new multipliers
        fn set_multipliers(&mut self, multipliers: Vec<f64>);
    }

    /// Ising model sampler implementation
    pub struct IsingCore {
        pub n: usize,
        pub coupling_mat: Vec<Vec<f64>>,
        pub multipliers: Vec<f64>,
        pub rng: StdRng,
        pub unitrng: Uniform<f64>,
    }

    impl IsingCore {
        /// Create a new Ising model sampler
        pub fn new(n: usize, multipliers: Vec<f64>, seed: Option<u64>) -> Self {
            let rng = match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };

            let unitrng = Uniform::new_inclusive(0.0, 1.0);
            let mut coupling_mat = vec![vec![0.0; n]; n];
            
            // Setup couplings by copying to a matrix (matching C++ logic)
            let mut counter = 0;
            for i in 0..(n-1) {
                for j in (i+1)..n {
                    if counter + n < multipliers.len() {
                        coupling_mat[i][j] = multipliers[counter + n];
                        coupling_mat[j][i] = multipliers[counter + n];
                    }
                    counter += 1;
                }
            }

            IsingCore {
                n,
                coupling_mat,
                multipliers,
                rng,
                unitrng,
            }
        }

        /// Set coupling matrix (pairwise interactions)
        pub fn set_coupling_matrix(&mut self, matrix: Vec<Vec<f64>>) {
            self.coupling_mat = matrix;
        }

        /// Get coupling matrix
        pub fn get_coupling_matrix(&self) -> &Vec<Vec<f64>> {
            &self.coupling_mat
        }
    }

    impl SamplerCore for IsingCore {
        fn calc_energy(&self, config: &[i32]) -> f64 {
            let mut energy = 0.0;
            let mut counter = 0;
            
            // Match C++ logic exactly: field terms first, then couplings
            for i in 0..(self.n - 1) {
                // Field terms: -sum(h_i * s_i)
                energy -= self.multipliers[i] * config[i] as f64;
                
                // Coupling terms: -sum(J_ij * s_i * s_j)
                for j in (i+1)..self.n {
                    energy -= self.multipliers[counter + self.n] * config[i] as f64 * config[j] as f64;
                    counter += 1;
                }
            }
            
            // Last field term (for the last spin)
            energy -= self.multipliers[self.n - 1] * config[self.n - 1] as f64;
            
            energy
        }

        fn metropolis_step(&mut self, config: &mut Vec<i32>) -> f64 {
            let site_uniform = Uniform::new_inclusive(0, self.n - 1);
            let site = site_uniform.sample(&mut self.rng);
            
            // Flip the spin
            config[site] *= -1;
            
            // Calculate energy difference analytically (matching C++ logic)
            let mut delta_e = -2.0 * self.multipliers[site] * config[site] as f64;
            
            // Coupling contributions
            for i in 0..self.n {
                delta_e -= 2.0 * self.coupling_mat[site][i] * config[site] as f64 * config[i] as f64;
            }
            
            // Metropolis acceptance criterion
            if delta_e > 0.0 && self.unitrng.sample(&mut self.rng) > (-delta_e).exp() {
                // Reject the move: flip back
                config[site] *= -1;
                0.0
            } else {
                delta_e
            }
        }

        fn init_config(&mut self) -> Vec<i32> {
            let uniform = Uniform::new_inclusive(-1, 1);
            (0..self.n).map(|_| uniform.sample(&mut self.rng)).collect()
        }

        fn system_size(&self) -> usize {
            self.n
        }

        fn get_multipliers(&self) -> &[f64] {
            &self.multipliers
        }

        fn set_multipliers(&mut self, multipliers: Vec<f64>) {
            self.multipliers = multipliers;
        }
    }

    /// 3-state Potts model sampler implementation
    pub struct Potts3Core {
        pub n: usize,
        pub coupling_mat: Vec<Vec<f64>>,
        pub multipliers: Vec<f64>,
        pub rng: StdRng,
        pub state_rng: Uniform<usize>,
        pub unitrng: Uniform<f64>,
    }

    impl Potts3Core {
        /// Create a new 3-state Potts model sampler
        pub fn new(n: usize, multipliers: Vec<f64>, seed: Option<u64>) -> Self {
            let rng = match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };

            let state_rng = Uniform::new_inclusive(1, 2); // For state transitions: 1 or 2
            let unitrng = Uniform::new_inclusive(0.0, 1.0);
            let mut coupling_mat = vec![vec![0.0; n]; n];
            
            // Setup couplings by copying to a matrix (matching C++ logic)
            let mut counter = 0;
            for i in 0..(n-1) {
                for j in (i+1)..n {
                    if counter + 3*n < multipliers.len() {
                        coupling_mat[i][j] = multipliers[counter + 3*n];
                        coupling_mat[j][i] = multipliers[counter + 3*n];
                    }
                    counter += 1;
                }
            }

            Potts3Core {
                n,
                coupling_mat,
                multipliers,
                rng,
                state_rng,
                unitrng,
            }
        }

        /// Set coupling matrix (pairwise interactions)
        pub fn set_coupling_matrix(&mut self, matrix: Vec<Vec<f64>>) {
            self.coupling_mat = matrix;
        }

        /// Get coupling matrix
        pub fn get_coupling_matrix(&self) -> &Vec<Vec<f64>> {
            &self.coupling_mat
        }
    }

    impl SamplerCore for Potts3Core {
        fn calc_energy(&self, config: &[i32]) -> f64 {
            let mut energy = 0.0;
            let mut counter = 0;
            
            // Match C++ logic exactly: 3 field terms per spin, then couplings
            for i in 0..(self.n - 1) {
                // Field terms: different for each state (0, 1, 2)
                if config[i] == 0 {
                    energy -= self.multipliers[i];
                } else if config[i] == 1 {
                    energy -= self.multipliers[i + self.n];
                } else {
                    energy -= self.multipliers[i + 2*self.n];
                }
                
                // Coupling terms: delta function for same states
                for j in (i+1)..self.n {
                    if config[i] == config[j] {
                        energy -= self.multipliers[counter + 3*self.n];
                    }
                    counter += 1;
                }
            }
            
            // Last field term (for the last spin)
            if config[self.n - 1] == 0 {
                energy -= self.multipliers[self.n - 1];
            } else if config[self.n - 1] == 1 {
                energy -= self.multipliers[2*self.n - 1];
            } else {
                energy -= self.multipliers[3*self.n - 1];
            }
            
            energy
        }

        fn metropolis_step(&mut self, config: &mut Vec<i32>) -> f64 {
            let site_uniform = Uniform::new_inclusive(0, self.n - 1);
            let site = site_uniform.sample(&mut self.rng);
            let old_state = config[site];
            
            // Try new state using C++ logic: (staterng(rd)+oState)%3
            config[site] = ((self.state_rng.sample(&mut self.rng) + old_state as usize) % 3) as i32;
            
            // Calculate energy difference analytically (matching C++ logic)
            let mut delta_e = 0.0;
            
            // Field contributions (matching C++ indexing)
            delta_e += self.multipliers[old_state as usize * self.n + site];  // remove old field
            delta_e -= self.multipliers[config[site] as usize * self.n + site];  // add new field
            
            // Coupling contributions
            for i in 0..self.n {
                if old_state == config[i] {
                    delta_e += self.coupling_mat[site][i];
                }
                if config[site] == config[i] {
                    delta_e -= self.coupling_mat[site][i];
                }
            }
            
            // Metropolis acceptance criterion
            if delta_e > 0.0 && self.unitrng.sample(&mut self.rng) > (-delta_e).exp() {
                // Reject the move: restore old state
                config[site] = old_state;
                0.0
            } else {
                delta_e
            }
        }

        fn init_config(&mut self) -> Vec<i32> {
            let init_rng = Uniform::new_inclusive(0, 2); // 3 states: 0, 1, 2
            (0..self.n).map(|_| init_rng.sample(&mut self.rng) as i32).collect()
        }

        fn system_size(&self) -> usize {
            self.n
        }

        fn get_multipliers(&self) -> &[f64] {
            &self.multipliers
        }

        fn set_multipliers(&mut self, multipliers: Vec<f64>) {
            self.multipliers = multipliers;
        }
    }

    /// Generic sampling algorithm that works with any SamplerCore implementation
    pub fn generate_samples<T: SamplerCore>(
        sampler: &mut T,
        n_samples: usize,
        burn_in: usize,
        steps_per_sample: usize,
        verbose: bool,
    ) -> Vec<Vec<i32>> {
        let mut samples = Vec::new();
        
        // Initialize random configuration
        let mut config = sampler.init_config();
        
        // Burn-in phase
        for _ in 0..burn_in {
            for _ in 0..steps_per_sample {
                sampler.metropolis_step(&mut config);
            }
        }
        
        // Generate samples
        for i in 0..n_samples {
            for _ in 0..steps_per_sample {
                sampler.metropolis_step(&mut config);
            }
            
            samples.push(config.clone());
            
            if verbose && i % 100 == 0 {
                println!("Generated {} samples", i + 1);
            }
        }
        
        samples
    }

    /// Calculate sample means
    pub fn calculate_means(samples: &[Vec<i32>], n_vars: usize) -> Vec<f64> {
        if samples.is_empty() {
            return vec![0.0; n_vars];
        }
        
        let mut means = vec![0.0; n_vars];
        let n_samples = samples.len() as f64;
        
        for sample in samples {
            for (i, &value) in sample.iter().enumerate() {
                if i < n_vars {
                    means[i] += value as f64;
                }
            }
        }
        
        for mean in &mut means {
            *mean /= n_samples;
        }
        
        means
    }

    /// Calculate sample correlations (pairwise)
    pub fn calculate_correlations(samples: &[Vec<i32>], n_vars: usize) -> Vec<Vec<f64>> {
        if samples.is_empty() {
            return vec![vec![0.0; n_vars]; n_vars];
        }
        
        let mut correlations = vec![vec![0.0; n_vars]; n_vars];
        let n_samples = samples.len() as f64;
        
        for sample in samples {
            for i in 0..n_vars {
                for j in 0..n_vars {
                    if i < sample.len() && j < sample.len() {
                        correlations[i][j] += (sample[i] as f64) * (sample[j] as f64);
                    }
                }
            }
        }
        
        for i in 0..n_vars {
            for j in 0..n_vars {
                correlations[i][j] /= n_samples;
            }
        }
        
        correlations
    }

    /// Calculate energy statistics from samples
    pub fn calculate_energy_stats<T: SamplerCore>(
        sampler: &T,
        samples: &[Vec<i32>],
    ) -> (f64, f64, f64) {
        if samples.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        
        let energies: Vec<f64> = samples.iter()
            .map(|sample| sampler.calc_energy(sample))
            .collect();
        
        let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;
        let variance = energies.iter()
            .map(|&e| (e - mean_energy).powi(2))
            .sum::<f64>() / energies.len() as f64;
        let std_dev = variance.sqrt();
        
        (mean_energy, variance, std_dev)
    }
}

/// Utility functions for statistical analysis
pub mod utils {
    use super::*;

    /// Calculate autocorrelation function
    pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
        if data.is_empty() {
            return Vec::new();
        }
        
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        
        if variance == 0.0 {
            return vec![1.0; max_lag.min(n)];
        }
        
        let mut autocorr = Vec::new();
        for lag in 0..max_lag.min(n) {
            let mut sum = 0.0;
            for i in 0..(n - lag) {
                sum += (data[i] - mean) * (data[i + lag] - mean);
            }
            autocorr.push(sum / ((n - lag) as f64 * variance));
        }
        
        autocorr
    }

    /// Estimate integrated autocorrelation time
    pub fn integrated_autocorr_time(data: &[f64], max_lag: Option<usize>) -> f64 {
        let max_lag = max_lag.unwrap_or(data.len() / 4);
        let autocorr = autocorrelation(data, max_lag);
        
        let mut tau_int = 0.5; // Start with 0.5 (uncorrelated)
        for (i, &corr) in autocorr.iter().enumerate() {
            if corr > 0.0 {
                tau_int += corr;
            } else {
                break;
            }
        }
        
        tau_int
    }

    /// Calculate effective sample size
    pub fn effective_sample_size(data: &[f64]) -> f64 {
        let tau_int = integrated_autocorr_time(data, None);
        data.len() as f64 / (2.0 * tau_int + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ising_energy() {
        let ising = samplers::IsingCore::new(2, vec![0.1, 0.2], Some(42));
        let config = vec![1, -1];
        let energy = ising.calc_energy(&config);
        assert!((energy - (-0.1 + 0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_potts_energy() {
        let potts = samplers::Potts3Core::new(2, vec![0.1, 0.2], Some(42));
        let config = vec![0, 1];
        let energy = potts.calc_energy(&config);
        assert!((energy - (-0.1 - 0.2)).abs() < 1e-10);
    }

    #[test]
    fn test_sample_means() {
        let samples = vec![vec![1, -1], vec![-1, 1], vec![1, 1]];
        let means = samplers::calculate_means(&samples, 2);
        assert!((means[0] - 1.0/3.0).abs() < 1e-10);
        assert!((means[1] - 1.0/3.0).abs() < 1e-10);
    }
}
