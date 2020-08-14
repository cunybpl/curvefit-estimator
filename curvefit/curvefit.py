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
        loss: str='linear', 
        method: str='trf', 
        jac: Union[str, Callable[[np.array, Any], np.array], None ]=None, 
        lsq_kwargs: dict={}
        ) -> None: 
        """ Wraps the scipy.optimize.curve_fit function for non-linear least squares. The curve_fit function is itself a wrapper around 
        scipy.optimize.leastsq and/or scipy.optimize.least_squares that aims to simplfy some of the calling mechanisms. An entrypoint 
        to kwargs for these lower level method is provided by the lsq_kwargs dictionary. These include parameters such as 
        ``f_scale`` and ``beta_guess`` which may need to be configured based on your problem/dataset. 

        On success, the curve_fit function will return a tuple of the optimized parameters to the function (popt) as well as the estimated
        covariance of these parameters. These values are used in the predict method and can be accessed after the model has been fit 
        as ``model.popt_`` and ``model.pcov_``. 

        It is best to refer to these docs to understand the methods being wrapped:
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

        Args:
            model_func (Callable[[], np.array], optional): The function to model. Defaults to None.
            bounds (Union[ Tuple[np.dtype, np.dtype], List[Tuple[np.dtype, np.dtype]] ], optional): 
                Search bounds either calculated in advance or at runtime. 
                Defaults to (-np.inf, np.inf).
            loss (str, optional): [description]. Defaults to 'linear'.
            beta_guess (Optional[Any], optional): [description]. Defaults to None.
            method (str, optional): Optimization method. Defaults to 'trf'.
            jac (Optional[ Callable[[np.array, Any], np.array] ], optional): Computes the jacobian matrix. Defaults to None.
            lsq_kwargs (dict, optional): Optional kwargs that are passed into the private 
                least_squares or leastsq functions refer to scipy docs. Defaults to {}.

        Raises:
            TypeError: Thrown if a model function is not provided.
        """
        if model_func is None:
            raise TypeError('Must provide a function to model.')

        self.model_func = model_func
        self.p0 = p0 
        self.bounds = bounds 
        self.loss = loss 
        self.method = method
        self.jac = jac
        self.lsq_kwargs = lsq_kwargs


    def fit(self, 
        X: np.array, 
        y: np.array=None, 
        sigma: Optional[np.array]=None, 
        absolute_sigma: bool=True) -> 'CurvefitEstimator':
        """ Fit X features to target y. 

        Refer to scipy.optimize.curve_fit docs for details on sigma values.

        Args:
            X (np.array): The feature matrix we are using to fit.
            y (np.array): The target array.
            sigma (Optional[np.array], optional): Determines uncertainty in the ydata. Defaults to None.
            absolute_sigma (bool, optional): Uses sigma in an absolute sense and reflects this in the pcov. Defaults to True.

        Returns:
            GeneralizedCurveFitEstimator: [description]
        """
        X, y = check_X_y(X, y)

        popt, pcov = optimize.curve_fit(f=self.model_func, 
            xdata=X, 
            ydata=y, 
            p0=self.p0, 
            method=self.method,
            sigma=sigma,
            absolute_sigma=absolute_sigma,
            loss=self.loss,
            bounds=self.bounds,
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
        X = check_array(X, ensure_2d=False)

        return self.model_func(X, *self.popt_) 