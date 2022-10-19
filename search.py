from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import get_scorer
import tensorflow as tf
import numpy as np
import joblib as jb

class GradientSearchCV:
    def __init__(
        self,
        param_list,
        estimator,
        *,
        n_iter=10,
        cv=3
        learning_rate=0.1,
        scoring=None
        early_stopping_rounds=None,
        n_jobs=1
    ):
    
    assert hasattr(param_list, '__iter__'), 'The parameter list must be an iterable.'
    self.param_list = param_list
    assert issubclass(estimator, BaseEstimator), 'The estimator must be a scikit-learn estimator.'
    self.estimator = estimator
    assert type(n_iter) is int, 'n_iter must be an integer.'
    self.n_iter = n_iter
    self.cv = cv
    assert learning_rate > 0 and type(learning_rate) in [int, float], 'The learning rate must be numerical and positive.'
    self.learning_rate = learning_rate
    if early_stopping_rounds is not None:
        assert type(early_stopping_rounds) is int and early_stopping_rounds > 0, 'The number of early stopping rounds must be a positive integer.'
    self.early_stopping_rounds = early_stopping_rounds
    assert not (eval_set is not None and validation_split is not None)
    self.scoring = scoring
    if type(scoring) is str:
        self._isloss = 'neg_' in scoring
    self.n_jobs = n_jobs 
    
    def fit(X, y, fit_params):
        model = clone(estimator).set_params(**fit_params)
        param_dict = {param: np.random.normal() for param in self.param_list}
        with tf.GradientTape(persistent=True) as gt:
            loss_list = []
            param_matrix = []
            consecutive_count = 0
            parameters = tf.Variable(list(param_dict.values))
            for _ in range(n_iter):
                gt.watch(parameters)
                model_with_params = model.set_params(**param_dict)
                model_with_params.fit(X_train, y_train)
                if self._isloss:
                    loss = tf.Variable(-1*cross_val_score(model_with_params, X, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs))
                else:
                    loss = tf.Variable(minmax_scale(-1*cross_val_score(model_with_params, X, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs)))
                diff = -self.learning_rate + gt.gradient(loss, parameters)
                parameters += diff
                param_matrix[_] = parameters
                loss_list[_] = loss.numpy()
                if early_stopping_rounds is not None:
                    if _ != 0 and loss_list[_-1] <= loss.numpy().mean():
                        consecutive_count += 1
                    else:
                        consecutive_count = 0
                    if consecutive_count == self.early_stopping_rounds:
                        break
        param_matrix = np.array(param_matrix)
        means = param_matrix.mean(axis=1)
        best_param_index = np.argmin(means)
        self.best_params_ = {name: value for name, value in zip(self.param_list, param_matrix[best_param_index])}
        return self
            
