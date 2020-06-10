"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator,
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class GeneralizedCurveFitEstimator(BaseEstimator, RegressorMixin):

    def __init__(self, model_func=None, bounds=None, loss='linear', f_scale=1.0, beta_guess=None, method ='trf', sigma=None): 
        self.model_func = model_func 
        self.bounds = bounds 
        self.loss = loss 
        self.f_scale = f_scale  
        self.beta_guess = beta_guess
        self.sigma = sigma
        self.method = method


    def fit(self, X, y):
        
        popt, pcov = curve_fit(f = self.model_func, xdata=X, ydata=y, 
            p0 = self.beta_guess, 
            method = self.method,
            sigma=self.sigma,
            loss=self.loss,
            f_scale=self.f_scale, 
            bounds=self.bounds(X) # XXX <--- [(min(beta), max(beta)), ()]
            )
        self.coef_ = popt
        self.cov = pcov
        return self 

    def predict(self, X, ensure_2d=False):
        X = check_array(X, ensure_2d=ensure_2d)
        return self.model_func(X, *self.coef_) 