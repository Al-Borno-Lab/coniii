#!/usr/bin/env python3
"""
Clean build artifacts and temporary files.
Run this script to clean up build artifacts from the project.
"""

import os
import shutil
import glob

def clean_build_artifacts():
    """Clean up build artifacts and temporary files."""
    
    # Directories to clean
    dirs_to_clean = [
        "build/",
        "target/",
        "dist/",
        "*.egg-info/",
        "coniii/__pycache__/",
        "coniii/*/__pycache__/",
    ]
    
    # Files to clean
    files_to_clean = [
        "coniii/*.so",
        "coniii/*.dylib", 
        "coniii/*.dll",
        "coniii/*.pyd",
        "*.pyc",
        "**/*.pyc",
    ]
    
    print("🧹 Cleaning build artifacts...")
    
    # Clean directories
    for pattern in dirs_to_clean:
        for path in glob.glob(pattern, recursive=True):
            if os.path.exists(path):
                print(f"  Removing directory: {path}")
                shutil.rmtree(path)
    
    # Clean files
    for pattern in files_to_clean:
        for path in glob.glob(pattern, recursive=True):
            if os.path.exists(path):
                print(f"  Removing file: {path}")
                os.remove(path)
    
    print("✅ Cleanup complete!")

if __name__ == "__main__":
    clean_build_artifacts()
