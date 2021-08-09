import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose

from curvefit import CurvefitEstimator
import unittest

tc = unittest.TestCase()


def test_curvefit_estimator_against_scipy_example():
    # Use the scipy example to fit a function. 

    def func(x, a, b, c): 
        return a * np.exp(-b * x) + c 

    np.random.seed(1729)

    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise

    estimator = CurvefitEstimator(model_func=func)
    estimator.fit(xdata.reshape(-1, 1), ydata, squeeze_X=True)
    expected = [
        2.5542373783987986, 
        1.3519104048478578, 
        0.47450642597289794
    ]
    assert list(expected) == list(estimator.popt_)

    predicted_res = estimator.predict(xdata)
    print(predicted_res)
    

def test_curvefit_estimator_with_pipeline_api():
    raise NotImplementedError 


    