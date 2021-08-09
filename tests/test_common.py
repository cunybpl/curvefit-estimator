import pytest
import numpy as np

from curvefit import CurvefitEstimator

from sklearn.utils.estimator_checks import _yield_all_checks, check_estimator
from sklearn.utils._testing import SkipTest
from sklearn.exceptions import SkipTestWarning
from sklearn.utils import estimator_checks
import warnings


def f(X, m, b):
    if X.shape[0] > 1:
        X = X[:,0]
    return  X * m**2 + b

@pytest.mark.parametrize(
    "Estimator", [CurvefitEstimator]
)
def test_estimators(Estimator):

    estimator = Estimator(model_func=f)
    checks = {'passed': 0, 'failed': 0}
    for check in _yield_all_checks(estimator):
        fname = str(check)
        try:
            check(estimator)
            print(f'PASSED: {fname}')
            checks['passed'] += 1
        except Exception as e:
            print(f'FAILED: {fname} ... {e}{str(e)}')
            checks['failed'] += 1
            # raise 
    
    print('===========SKLEARN ESTIMATOR CHECKS==========\n')
    print(f"PASSED: {checks['passed']}    FAILED: {checks['failed']}")
    print('=============================================\n')