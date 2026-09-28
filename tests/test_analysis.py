import pytest
import pandas as pd
import numpy as np

# Import core modules
from space_project.pc_analytical import compute_collision_probability
from space_project.propagate import transform_coordinates
from space_project.upload_handler import parse_custom_tle_file

def test_analytical_collision_probability():
    """Validates the analytical collision probability (Pc) math handles valid inputs."""
    # Using compute_collision_probability from pc_analytical
    result = compute_collision_probability(
        miss_distance_km=0.5,
        relative_velocity_kms=7.5,
        sigma_r_a=10.0,
        sigma_r_b=10.0,
        sigma_t_a=10.0,
        sigma_t_b=10.0,
        hard_body_radius_a_m=5.0,
        hard_body_radius_b_m=5.0,
    )
    
    assert "pc" in result
    pc = result["pc"]
    assert isinstance(pc, float)
    assert 0.0 <= pc <= 1.0

def test_sgp4_coordinate_transformations():
    """Validates the SGP4 coordinate transformations yield expected types and bounds."""
    state_vector = [7000.0, 0.0, 0.0, 0.0, 7.5, 0.0]
    epoch = 2459000.5
    
    x, y, z = transform_coordinates(state_vector, epoch)
    
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert isinstance(z, float)
    assert x == 7000.0

def test_custom_csv_upload_parser():
    """Validates the custom CSV/TLE upload parser mapping logic."""
    # Testing the parse_custom_tle_file since it's the upload handler available
    mock_tle = b"ISS\n1 25544U 98067A   21123.54321098  .00001234  00000-0  12345-4 0  9999\n2 25544  51.6432 123.4567 0001234  45.6789  98.7654 15.50123456123456"
    
    records = parse_custom_tle_file(mock_tle, "test_file.txt")
    
    assert len(records) == 1
    assert "norad_id" in records[0]
    assert records[0]["norad_id"] == 25544
