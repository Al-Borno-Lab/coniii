#!/usr/bin/env python3
"""
Intensive stress test for the Rust implementation.
Tests extreme conditions and system sizes.
"""

import time
import numpy as np
from coniii import Ising, Potts3
import psutil
import os
import threading
import concurrent.futures
from typing import List, Dict

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def stress_test_extreme_sizes():
    """Test with extremely large system sizes."""
    print("🔥 EXTREME SIZE STRESS TEST")
    print("=" * 50)
    
    extreme_sizes = [200, 500, 1000, 2000]
    results = []
    
    for n in extreme_sizes:
        print(f"\nTesting system size: {n}")
        
        try:
            # Create massive system
            n_couplings = n * (n - 1) // 2
            print(f"  Total parameters: {n + n_couplings:,}")
            
            # Generate multipliers
            multipliers = np.random.normal(0, 0.1, n + n_couplings).tolist()
            
            memory_before = get_memory_usage()
            start_time = time.time()
            
            # Create and test
            ising = Ising(n, multipliers, seed=42)
            
            # Test energy calculation
            config = np.random.choice([-1, 1], n).tolist()
            energy = ising.calc_e(config)
            
            # Test Metropolis sampling
            ising.sample_metropolis(steps=100)
            
            # Test sample generation (small sample for large systems)
            n_samples = min(50, 1000 // n)  # Adaptive sample size
            ising.generate_sample(n_samples=n_samples, burn_in=10, steps=2, verbose=False)
            
            end_time = time.time()
            memory_after = get_memory_usage()
            
            time_taken = end_time - start_time
            memory_used = memory_after - memory_before
            
            results.append({
                'system_size': n,
                'parameters': n + n_couplings,
                'time_taken': time_taken,
                'memory_used': memory_used,
                'energy': energy,
                'success': True
            })
            
            print(f"  ✅ Success: {time_taken:.3f}s, {memory_used:.1f}MB, Energy: {energy:.3f}")
            
        except Exception as e:
            results.append({
                'system_size': n,
                'parameters': n + n_couplings if 'n_couplings' in locals() else 0,
                'time_taken': None,
                'memory_used': None,
                'energy': None,
                'success': False,
                'error': str(e)
            })
            print(f"  ❌ Failed: {e}")
    
    return results

def stress_test_massive_samples():
    """Test with massive sample counts."""
    print("\n🔥 MASSIVE SAMPLES STRESS TEST")
    print("=" * 50)
    
    n = 10  # Fixed system size
    n_couplings = n * (n - 1) // 2
    multipliers = np.random.normal(0, 0.1, n + n_couplings).tolist()
    
    massive_samples = [10000, 50000, 100000, 500000]
    results = []
    
    for n_samples in massive_samples:
        print(f"\nTesting {n_samples:,} samples...")
        
        try:
            memory_before = get_memory_usage()
            start_time = time.time()
            
            ising = Ising(n, multipliers, seed=42)
            ising.generate_sample(n_samples=n_samples, burn_in=100, steps=5, verbose=False)
            
            end_time = time.time()
            memory_after = get_memory_usage()
            
            time_taken = end_time - start_time
            memory_used = memory_after - memory_before
            throughput = n_samples / time_taken
            
            results.append({
                'n_samples': n_samples,
                'time_taken': time_taken,
                'memory_used': memory_used,
                'throughput': throughput,
                'success': True
            })
            
            print(f"  ✅ Success: {time_taken:.3f}s, {memory_used:.1f}MB, "
                  f"{throughput:,.0f} samples/s")
            
        except Exception as e:
            results.append({
                'n_samples': n_samples,
                'time_taken': None,
                'memory_used': None,
                'throughput': None,
                'success': False,
                'error': str(e)
            })
            print(f"  ❌ Failed: {e}")
    
    return results

def stress_test_concurrent():
    """Test concurrent access to multiple samplers."""
    print("\n🔥 CONCURRENT ACCESS STRESS TEST")
    print("=" * 50)
    
    def worker(worker_id: int, n: int, n_samples: int) -> Dict:
        """Worker function for concurrent testing."""
        try:
            n_couplings = n * (n - 1) // 2
            multipliers = np.random.normal(0, 0.1, n + n_couplings).tolist()
            
            start_time = time.time()
            
            # Create both Ising and Potts3 samplers
            ising = Ising(n, multipliers, seed=42 + worker_id)
            potts = Potts3(n, multipliers[:3*n + n_couplings], seed=42 + worker_id)
            
            # Generate samples
            ising.generate_sample(n_samples=n_samples, burn_in=50, steps=3, verbose=False)
            potts.generate_sample(n_samples=n_samples, burn_in=50, steps=3, verbose=False)
            
            end_time = time.time()
            
            return {
                'worker_id': worker_id,
                'time_taken': end_time - start_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'worker_id': worker_id,
                'time_taken': None,
                'success': False,
                'error': str(e)
            }
    
    # Test with different numbers of concurrent workers
    worker_counts = [2, 4, 8, 16]
    n = 8
    n_samples = 1000
    
    for num_workers in worker_counts:
        print(f"\nTesting {num_workers} concurrent workers...")
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i, n, n_samples) for i in range(num_workers)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"  ✅ {len(successful)}/{num_workers} workers successful")
        print(f"  Total time: {total_time:.3f}s")
        if failed:
            print(f"  ❌ {len(failed)} workers failed: {[f['error'] for f in failed]}")

def stress_test_memory_intensive():
    """Test memory-intensive operations."""
    print("\n🔥 MEMORY INTENSIVE STRESS TEST")
    print("=" * 50)
    
    # Create many samplers to test memory usage
    samplers = []
    n = 20
    
    try:
        for i in range(100):  # Create 100 samplers
            n_couplings = n * (n - 1) // 2
            multipliers = np.random.normal(0, 0.1, n + n_couplings).tolist()
            
            ising = Ising(n, multipliers, seed=42 + i)
            potts = Potts3(n, multipliers[:3*n + n_couplings], seed=42 + i)
            
            samplers.append((ising, potts))
            
            if i % 20 == 0:
                memory_used = get_memory_usage()
                print(f"  Created {i+1} samplers, Memory: {memory_used:.1f}MB")
        
        print(f"  ✅ Successfully created {len(samplers)} samplers")
        
        # Test operations on all samplers
        print("  Testing operations on all samplers...")
        start_time = time.time()
        
        for i, (ising, potts) in enumerate(samplers):
            # Generate samples
            ising.generate_sample(n_samples=100, burn_in=10, steps=2, verbose=False)
            potts.generate_sample(n_samples=100, burn_in=10, steps=2, verbose=False)
            
            if i % 20 == 0:
                print(f"    Processed {i+1}/{len(samplers)} samplers")
        
        end_time = time.time()
        total_time = end_time - start_time
        memory_used = get_memory_usage()
        
        print(f"  ✅ All operations completed in {total_time:.3f}s")
        print(f"  Final memory usage: {memory_used:.1f}MB")
        
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")

def stress_test_energy_calculations():
    """Stress test energy calculations with many configurations."""
    print("\n🔥 ENERGY CALCULATION STRESS TEST")
    print("=" * 50)
    
    n = 15
    n_couplings = n * (n - 1) // 2
    multipliers = np.random.normal(0, 0.1, n + n_couplings).tolist()
    
    ising = Ising(n, multipliers, seed=42)
    potts = Potts3(n, multipliers[:3*n + n_couplings], seed=42)
    
    # Generate many random configurations
    n_configs = 100000
    print(f"Testing {n_configs:,} energy calculations...")
    
    # Ising configurations
    ising_configs = [np.random.choice([-1, 1], n).tolist() for _ in range(n_configs)]
    
    # Potts3 configurations
    potts_configs = [np.random.choice([0, 1, 2], n).tolist() for _ in range(n_configs)]
    
    # Benchmark Ising energy calculations
    print("  Testing Ising energy calculations...")
    start_time = time.time()
    ising_energies = [ising.calc_e(config) for config in ising_configs]
    ising_time = time.time() - start_time
    
    # Benchmark Potts3 energy calculations
    print("  Testing Potts3 energy calculations...")
    start_time = time.time()
    potts_energies = [potts.calc_e(config) for config in potts_configs]
    potts_time = time.time() - start_time
    
    print(f"  ✅ Ising: {ising_time:.3f}s total, {ising_time/n_configs*1000:.3f}ms per calc")
    print(f"  ✅ Potts3: {potts_time:.3f}s total, {potts_time/n_configs*1000:.3f}ms per calc")
    print(f"  Ising energy range: {min(ising_energies):.3f} to {max(ising_energies):.3f}")
    print(f"  Potts3 energy range: {min(potts_energies):.3f} to {max(potts_energies):.3f}")

def main():
    """Run all stress tests."""
    print("🚀 RUST IMPLEMENTATION INTENSIVE STRESS TEST")
    print("=" * 60)
    
    # Run all stress tests
    extreme_results = stress_test_extreme_sizes()
    massive_results = stress_test_massive_samples()
    stress_test_concurrent()
    stress_test_memory_intensive()
    stress_test_energy_calculations()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 STRESS TEST SUMMARY")
    print("=" * 60)
    
    # Extreme sizes summary
    successful_extreme = [r for r in extreme_results if r['success']]
    if successful_extreme:
        max_size = max(r['system_size'] for r in successful_extreme)
        max_params = max(r['parameters'] for r in successful_extreme)
        print(f"💪 Extreme sizes: Successfully tested up to {max_size} spins ({max_params:,} parameters)")
    
    # Massive samples summary
    successful_massive = [r for r in massive_results if r['success']]
    if successful_massive:
        max_samples = max(r['n_samples'] for r in successful_massive)
        max_throughput = max(r['throughput'] for r in successful_massive)
        print(f"💪 Massive samples: Successfully generated {max_samples:,} samples")
        print(f"💪 Peak throughput: {max_throughput:,.0f} samples/second")
    
    print("\n🎉 All stress tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()