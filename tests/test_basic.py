import pytest
import torch
from quark.facade import MasterQuark

def test_master_quark_initialization():
    """Verify that MasterQuark initializes correctly with default and custom parameters."""
    # Test Default Initialization
    model_default = MasterQuark()
    assert model_default.max_assets == 8
    assert model_default.objective_type == 'composite'
    
    # Test Custom Initialization
    model_custom = MasterQuark(
        objective_type='cvar',
        max_assets=10,
        lower_bound=0.01,
        upper_bound=0.50,
        num_fireflies=50,
        max_iterations=100
    )
    assert model_custom.objective_type == 'cvar'
    assert model_custom.max_assets == 10
    assert model_custom.lower_bound == 0.01
    assert model_custom.upper_bound == 0.50
    assert model_custom.num_fireflies == 50
    assert model_custom.max_iterations == 100
