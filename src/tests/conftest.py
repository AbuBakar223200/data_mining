
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

@pytest.fixture
def synthetic_data():
    """Generate synthetic dataframe mimicking UNSW-NB15 structure."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': range(1, n_samples + 1),
        'dur': np.random.exponential(1, n_samples),
        'proto': np.random.choice(['tcp', 'udp', 'arp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp', '-'], n_samples),
        'state': np.random.choice(['FIN', 'CON', 'INT'], n_samples),
        'spkts': np.random.randint(1, 100, n_samples),
        'dpkts': np.random.randint(1, 100, n_samples),
        'sbytes': np.random.randint(100, 10000, n_samples),
        'dbytes': np.random.randint(100, 10000, n_samples),
        'rate': np.random.random(n_samples) * 100,
        'sttl': np.random.randint(1, 64, n_samples),
        'dttl': np.random.randint(1, 64, n_samples),
        'sload': np.random.random(n_samples) * 1000,
        'dload': np.random.random(n_samples) * 1000,
        'sloss': np.random.randint(0, 10, n_samples),
        'dloss': np.random.randint(0, 10, n_samples),
        'sinpkt': np.random.random(n_samples),
        'dinpkt': np.random.random(n_samples),
        'sjit': np.random.random(n_samples),
        'djit': np.random.random(n_samples),
        'swin': np.random.randint(0, 255, n_samples),
        'stcpb': np.random.random(n_samples) * 10000,
        'dtcpb': np.random.random(n_samples) * 10000,
        'dwin': np.random.randint(0, 255, n_samples),
        'tcprtt': np.random.random(n_samples),
        'synack': np.random.random(n_samples),
        'ackdat': np.random.random(n_samples),
        'smean': np.random.randint(40, 1500, n_samples),
        'dmean': np.random.randint(40, 1500, n_samples),
        'trans_depth': np.random.randint(0, 5, n_samples),
        'response_body_len': np.random.randint(0, 1000, n_samples),
        'ct_srv_src': np.random.randint(1, 10, n_samples),
        'ct_state_ttl': np.random.randint(1, 6, n_samples),
        'ct_dst_ltm': np.random.randint(1, 10, n_samples),
        'ct_src_dport_ltm': np.random.randint(1, 10, n_samples),
        'ct_dst_sport_ltm': np.random.randint(1, 10, n_samples),
        'ct_dst_src_ltm': np.random.randint(1, 10, n_samples),
        'is_ftp_login': np.random.choice([0, 1], n_samples),
        'ct_ftp_cmd': np.random.randint(0, 2, n_samples),
        'ct_flw_http_mthd': np.random.randint(0, 2, n_samples),
        'ct_src_ltm': np.random.randint(1, 10, n_samples),
        'ct_srv_dst': np.random.randint(1, 10, n_samples),
        'is_sm_ips_ports': np.random.choice([0, 1], n_samples),
        
        # Leakage columns (to test dropping)
        'srcip': [f'192.168.1.{i}' for i in range(n_samples)],
        'dstip': [f'10.0.0.{i}' for i in range(n_samples)],
        'sport': np.random.randint(1024, 65535, n_samples),
        'dsport': np.random.randint(80, 8080, n_samples),
        'stime': range(1000, 1000 + n_samples),
        'ltime': range(1000, 1000 + n_samples),
        
        # Targets
        'attack_cat': np.random.choice(['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance'], n_samples),
        'label': np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def mock_config(tmp_path):
    """Generate a mock configuration dictionary."""
    return {
        'results_dir': str(tmp_path / "results"),
        'data': {
            # Paths handled by mocks usually, but we might test loading
            'train_path': str(tmp_path / "train.csv"),
            'test_path': str(tmp_path / "test.csv"),
            'target_binary': 'label',
            'target_multiclass': 'attack_cat',
            'drop_columns': ['id', 'srcip', 'dstip', 'sport', 'dsport', 'stime', 'ltime'],
            'categorical_columns': ['proto', 'service', 'state'],
            'numerical_columns': [] # Auto-detected usually
        },
        'experiments': {
            'n_seeds': 1,
            'n_jobs': 1,
            'tasks': ['binary', 'multi'],
            'models': ['rf'],
            'strategies': ['s0']
        },
        'random_state': 42
    }
