"""
Test the Rust integration functions.
"""

import pytest


def test_sum_as_string():
    """Test the Rust sum_as_string function."""
    from coniii import sum_as_string
    
    # Test basic functionality
    result = sum_as_string(5, 10)
    assert result == "15"
    assert isinstance(result, str)
    
    # Test with different values
    result2 = sum_as_string(0, 0)
    assert result2 == "0"
    
    result3 = sum_as_string(100, 200)
    assert result3 == "300"
    
    # Test with larger numbers
    result4 = sum_as_string(1000, 2000)
    assert result4 == "3000"


def test_rust_module_import():
    """Test that the Rust module can be imported."""
    try:
        from coniii import coniii as rust_module
        assert hasattr(rust_module, 'sum_as_string')
    except ImportError:
        pytest.skip("Rust module not available")


if __name__ == "__main__":
    pytest.main([__file__])
