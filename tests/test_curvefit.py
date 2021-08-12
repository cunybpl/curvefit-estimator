import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose

from curvefit.estimator import CurvefitEstimator
import unittest

from sklearn import pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


np.random.seed(1729)


def test_curvefit_estimator_against_scipy_example():
    # Use the scipy example to fit a function. 

    def func(x, a, b, c): 
        return (a * np.exp(-b * x) + c).squeeze() 

    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise


    estimator = CurvefitEstimator(model_func=func)
    estimator.fit(xdata.reshape(-1, 1), ydata)
    expected = [
        2.5542, 
        1.3519, 
        0.4745
    ]
    assert list(expected) == [round(p, 4) for p in list(estimator.popt_)]
    estimator.predict(xdata.reshape(-1,1))    # more or less just a smoke test
    

def test_curvefit_estimator_with_pipeline_api():
    
    # smoke test to make sure we are compatible with sklearn pipeline API
    def func(x, a, b, c): 
        return (a * np.exp(-b * x) + c).squeeze() 

    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise

    estimator = CurvefitEstimator(model_func=func)
    estimators = [('ct', StandardScaler()), ('f', estimator)]

    pipe = pipeline.Pipeline(estimators)
    pipe.fit(xdata.reshape(-1, 1), ydata)
    pipe.predict(xdata.reshape(-1,1)) 


def test_curvefit_grid_search(): 
    # smoketest that estimator works in a grid search

    def f(x, a, b, c): 
        return (a * np.exp(-b * x) + c).squeeze() 
    
    def g(x, a, b, c, d): 
        return (a * np.exp(b * x) + c - d).squeeze()

    xdata = np.linspace(0, 4, 50)
    y = f(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise

    params = {
        'model_func': [f, g],
    }

    search = GridSearchCV(CurvefitEstimator(), param_grid=params)
    search.fit(xdata.reshape(-1, 1), ydata)
    assert search.best_estimator_.name_ == 'f'


def test_callable_bounds(): 

    def f(x, a, b, c): 
        return (a * np.exp(-b * x) + c).squeeze() 

    xdata = np.linspace(0, 4, 50)
    y = f(xdata, 2.5, 1.3, 0.5)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise

    
    def bounds(X): 
        return (-np.inf, np.inf)

    est = CurvefitEstimator(model_func=f, bounds=bounds)
    est.fit(xdata.reshape(-1,1), ydata)