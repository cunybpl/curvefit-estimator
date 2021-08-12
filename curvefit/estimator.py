from typing import Callable, Tuple, Any, List, Optional, Union

import numpy as np

from scipy import optimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CurvefitEstimator(BaseEstimator, RegressorMixin):

    def __init__(self, 
        model_func: Callable[[], np.array]=None,
        p0: Optional[List[float]]=None, 
        bounds: Union[ Tuple[np.dtype, np.dtype], 
            List[Tuple[np.dtype, np.dtype]], 
            Callable[ [], List[Tuple[np.dtype, np.dtype]]] ]=(-np.inf, np.inf), 
        method: str='trf', 
        jac: Union[str, Callable[[np.array, Any], np.array], None ]=None, 
        lsq_kwargs: dict=None
        ) -> None:
        """Wraps the scipy.optimize.curve_fit function for non-linear least squares. The curve_fit function is itself a wrapper around 
        scipy.optimize.leastsq and/or scipy.optimize.least_squares that aims to simplfy some of the calling mechanisms. An entrypoint 
        to kwargs for these lower level method is provided by the lsq_kwargs dictionary.

        On success, the curve_fit function will return a tuple of the optimized parameters to the function (popt) as well as the estimated
        covariance of these parameters. These values are used in the predict method and can be accessed after the model has been fit 
            as ``model.popt_`` and ``model.pcov_``. 

        It is best to refer to these docs to understand the methods being wrapped:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        Args:
            model_func (Callable[[], np.array], optional): The function you wish to model. Defaults to None.
            p0 (Optional[List[float]], optional): The intial guess. Defaults to None.
            bounds (Union[ Tuple[np.dtype, np.dtype], List[Tuple[np.dtype, np.dtype]], 
                Callable[ [], List[Tuple[np.dtype, np.dtype]]] ], optional): Bounds for trf. Defaults to (-np.inf, np.inf).
            method (str, optional): The curve fit method. Defaults to 'trf'.
            jac (Union[str, Callable[[np.array, Any], np.array], None ], optional): The jacobian matrix. 
                If one is not provided then curve_fit will calculate it. Defaults to None.
            lsq_kwargs (dict, optional): Extra arguments for underlying lsq implementation. See `scipy.optimize.least_squares`. Defaults to None.
        """
        self.model_func = model_func
        self.p0 = p0 
        self.bounds = bounds 
        self.method = method
        self.jac = jac
        self.lsq_kwargs = lsq_kwargs if lsq_kwargs is not None else {}


    def fit(self, 
        X: np.array, 
        y: np.array=None, 
        sigma: Optional[np.array]=None, 
        absolute_sigma: bool=False) -> 'CurvefitEstimator':
        """ Fit X features to target y. 

        Refer to scipy.optimize.curve_fit docs for details on sigma values.

        Args:
            X (np.array): The feature matrix we are using to fit.
            y (np.array): The target array.
            sigma (Optional[np.array], optional): Determines uncertainty in the ydata. Defaults to None.
            absolute_sigma (bool, optional): Uses sigma in an absolute sense and reflects this in the pcov. Defaults to True.
            squeeze_1d: (bool, optional): Squeeze X into a 1 dimensional array for curve fitting. This is useful if you are fitting 
                a function with an X array and do not want to squeeze before it enters curve_fit. Defaults to True.
            
        Returns:
            GeneralizedCurveFitEstimator: self
        """
        # NOTE the user defined function should handle the neccesary array manipulation (squeeze, reshape etc.)
        X, y = check_X_y(X, y)  # pass the sklearn estimator dimensionality check
    
        if callable(self.bounds):  # we allow bounds to be a callable
            bounds = self.bounds(X)
        else:
            bounds = self.bounds

        popt, pcov = optimize.curve_fit(f=self.model_func, 
            xdata=X, 
            ydata=y, 
            p0=self.p0, 
            method=self.method,
            sigma=sigma,
            absolute_sigma=absolute_sigma, 
            bounds=bounds,
            jac=self.jac,
            **self.lsq_kwargs
            )

        self.popt_ = popt # set optimized parameters on the instance
        self.pcov_ = pcov # set optimzed covariances on the instance
        self.name_ = self.model_func.__name__ # name of func in case we are trying to fit multiple funcs in a Pipeline
        
        return self 


    def predict(self, X: np.array) -> np.array:
        """ Predict the target y values given X features using the best fit 
        model (model_func) and best fit model parameters (popt)

        Args:
            X (np.array): The X matrix 

        Returns:
            np.array: The predicted y values
        """
        check_is_fitted(self, ["popt_", "pcov_", "name_"])
        X = check_array(X)

        return self.model_func(X, *self.popt_) 