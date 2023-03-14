import time
import numpy as np

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, hinge_loss, accuracy_score

class Model:
    def train(self, x_train, y_train) -> float:
        pass

class LogisticRegressionModel:
    """
    LogisticRegressionModel is a wrapper for sklearn's LogisticRegression

    Parameters
    ----------
    solver : str
        The type of solver to use.
    tol : float
        The tolerance at which to stop iterating.
    
    Attributes
    ----------
    model : sklearn.linear_model.LogisticRegression
        The model to train and make predictions with.
    """
    def __init__(self, solver='newton-cholesky', tol=1e-4):
        self.model = LogisticRegression(solver=solver, tol=tol, random_state=0)

    def __str__(self) -> str:
        return "LogisticRegressionModel"

    def train(self, x_train, y_train) -> float:
        """Trains the model
        
        Parameters
        ----------
        x_train: ndarray
            The set of input values
        y_train: ndarray
            The correct outputs for the input values

        Returns
        -------
        time: float
            The time taken to train
        """
        start = time.time()
        self.model.fit(x_train, y_train)
        
        return time.time() - start

    def predict(self, x):
        return self.model.predict(x)
    
    def predict_proba(self, x):
        return self.model.predict_proba(x)
    
    def accuracy(self, x, y_true) -> float:
        score = self.model.score(x, y_true)
        
        return score
    
    def loss(self, x, y_true) -> float:
        y_pred = self.predict_proba(x)

        return log_loss(y_true, y_pred)
    
    def train_and_test(x_train, x_test, y_train, y_test) -> tuple:
        pass

"""
This contains code from Logistic-Regression-From-Scratch.
Licensing details are provided in the [README](./README.md)
It has been modified by me, Jakub Vogel.
"""
class RPropLogisticRegressionModel:
    def __init__(self, tol=1e-4, lr=0.01, etaminus=0.5, etaplus=1.2, minstep=1e-6, maxstep=50):
        self.losses = []
        self.train_accuracies = []
        self.tol = 1e-4
        self.lr = lr
        self.etaminus = etaminus
        self.etaplus = etaplus
        self.minstep = minstep
        self.maxstep = maxstep

    def __str__(self) -> str:
        return "RPropLogisticRegressionModel"

    def train(self, x_train, y_train, epochs=150) -> float:
        # Initialize LR weights and bias
        self.weights = np.zeros(x_train.shape[1])
        self.prev_weight_gradients = np.zeros(x_train.shape[1])
        self.prev_weight_steps = np.full(x_train.shape[1], self.lr)
        
        self.bias = 0
        self.prev_bias_gradient = self.lr
        self.prev_bias_step = self.lr

        prev_loss = None
        start = time.time()
        for i in range(epochs):
            prediction_probas = self.predict_proba(x_train)
            loss = self.loss(x_train, y_train)
            
            weight_gradients, bias_gradient = self._calculate_gradients(
                x_train, y_train, prediction_probas
            )

            self._update_weights(weight_gradients, bias_gradient)

            if prev_loss is None:
                prev_loss = loss
            elif (prev_loss - loss > 0) or (prev_loss - loss < self.tol):
                break

        return time.time() - start

    def predict(self, x):
        prediction_probas = self.predict_proba(x)

        return [1 if p > 0.5 else 0 for p in prediction_probas]

    def predict_proba(self, x):
        x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
        prediction_probas = self._sigmoid(x_dot_weights)

        return prediction_probas

    def accuracy(self, x, y_true):
        y_pred = self.predict(x)

        return accuracy_score(y_true, y_pred)
        
    def loss(self, x, y_true):
        y_pred = self.predict_proba(x)

        return log_loss(y_true, y_pred)
        


    def train_and_test(x_train, x_test, y_train, y_test) -> tuple:
        pass

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def _calculate_gradients(self, x, y_true, y_pred):
        # Gradient here is the derivative of binary cross entropy
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def _update_weights(self, weight_gradients, bias_gradient):
        # Update weights
        for j, curr_gradient in enumerate(weight_gradients):
            if self.prev_weight_gradients[j] * curr_gradient > 0: #same sign
                curr_step = self.prev_weight_steps[j] * self.etaplus
                curr_step = min(curr_step, self.maxstep)
            elif self.prev_weight_gradients[j] * curr_gradient < 0: #sign changed
                curr_step = self.prev_weight_steps[j] * self.etaminus
                curr_step = max(curr_step, self.minstep)
                curr_gradient = 0
            else:
                curr_step = self.prev_weight_steps[j]

            self.weights[j] += -1 * curr_step * np.sign(curr_gradient)
            self.prev_weight_steps[j] = curr_step
            self.prev_weight_gradients[j] = curr_gradient

        # Update bias
        curr_gradient = bias_gradient
        if self.prev_bias_gradient * curr_gradient > 0: #same sign
            curr_step = self.prev_bias_step * self.etaplus
            curr_step = min(curr_step, self.maxstep)
        elif self.prev_bias_gradient * curr_gradient < 0: #sign changed
            curr_step = self.prev_bias_step * self.etaminus
            curr_step = max(curr_step, self.minstep)
            curr_gradient = 0
        else:
            curr_step = self.prev_bias_step

class SVMModel:
    def __init__(self):
        self.model = svm.SVC()

    def __str__(self) -> str:
        return "SVMModel"

    def train(self, x_train, y_train) -> None:
        start = time.time()
        self.model.fit(x_train, y_train)
        
        return time.time() - start

    def predict(self, x):
        return self.model.predict(x)
    
    def accuracy(self, x, y_true):
        y_pred = self.predict(x)

        return accuracy_score(y_true, y_pred)
        
    def loss(self, x, y_true):
        y_pred = self.model.decision_function(x)

        return hinge_loss(y_true, y_pred)