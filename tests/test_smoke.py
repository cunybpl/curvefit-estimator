import toml 
import pathlib 

import curvefit_estimator


def test_versions_in_sync(): 

    # checks that toml and _version are the same to avoid packaging errors
    p = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(p, 'r') as f: 
        pyproject = toml.load(f)
        version = pyproject['tool']['poetry']['version']
    
    assert version == curvefit_estimator.__version__ 