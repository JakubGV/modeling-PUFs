import time
import numpy as np

from abc import ABC, abstractmethod

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, hinge_loss, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

def timeit(func):
    """Decorator that runs the passed function and returns the time taken to run
        
    Returns
    -------
    time: float
        The time taken for the function to run
    """
    def inner(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)

        return time.time() - start
    return inner

class Model(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass
    
    @abstractmethod
    def train(self, x_train, y_train) -> None:
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
        pass

    @abstractmethod
    def predict(self, x):
        """Predicts using the trained model
        
        Parameters
        ----------
        x: ndarray
            The set of input values

        Returns
        -------
        predictions: ndarray
            The predicted classifications (i.e., 0 or 1)
        """
        pass

    @abstractmethod
    def accuracy(self, x, y_true) -> float:
        """Calculate the accuracy of the model
        
        Parameters
        ----------
        x: ndarray
            The set of input values
        y_true: ndarray
            The correct outputs for the input values

        Returns
        -------
        accuracy: float
            The accuracy of the predicted outputs based on the true outputs
        """
        pass

    @abstractmethod
    def loss(self, x, y_true) -> float:
        """Calculate the loss of the model
        
        Parameters
        ----------
        x: ndarray
            The set of input values
        y_true: ndarray
            The correct outputs for the input values

        Returns
        -------
        loss: float
            The loss of the predicted outputs based on the true outputs
        """
        pass

class LogisticRegressionModel(Model):
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
        super().__init__()
        self.model = LogisticRegression(solver=solver, tol=tol, random_state=0)

    def __str__(self) -> str:
        return "LogisticRegressionModel"

    @timeit
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)
    
    def predict_proba(self, x):
        """Predict the probabilities of the output using the trained model
        
        Parameters
        ----------
        x: ndarray
            The set of input values

        Returns
        -------
        predictions: ndarray
            The predicted probabilities (i.e., continious from 0 to 1)
        """
        return self.model.predict_proba(x)
    
    def accuracy(self, x, y_true) -> float:
        score = self.model.score(x, y_true)
        
        return score
    
    def loss(self, x, y_true) -> float:
        y_pred = self.predict_proba(x)

        return log_loss(y_true, y_pred)

"""
This contains code from Logistic-Regression-From-Scratch.
Licensing details are provided in the [README](./README.md)
It has been modified by me, Jakub Vogel.
"""
class RPropLogisticRegressionModel(Model):
    """
    RPropLogisticRegressionModel is an implementation of logistic regression from scratch using RProp gradient descent
    
    Parameters
    ----------
    tol : float
        The tolerance, terminates training if the loss decreases by less than the tolerance
    lr : float
        The learning rate, sets the initial steps and bias gradient
    etaminus: float
        How much to multiplicatively decrease the weights
    etaplus: float
        How much to multiplicatively increase the weights
    minstep: float
        How much we can minimally decrease the weights
    maxstep: float
        How much we can maximally increase the weights
    """
    def __init__(self, tol=1e-4, lr=0.01, etaminus=0.5, etaplus=1.2, minstep=1e-6, maxstep=50):
        self.losses = []
        self.train_accuracies = []
        self.tol = tol
        self.lr = lr
        self.etaminus = etaminus
        self.etaplus = etaplus
        self.minstep = minstep
        self.maxstep = maxstep

    def __str__(self) -> str:
        return "RPropLogisticRegressionModel"

    @timeit
    def train(self, x_train, y_train, epochs=300):
        """Loosely based on PyTorch RProp [pseudocode](https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html)"""
        # Initialize LR weights and bias
        self.weights = np.zeros(x_train.shape[1])
        self.prev_weight_gradients = np.zeros(x_train.shape[1])
        self.prev_weight_steps = np.full(x_train.shape[1], self.lr)
        
        self.bias = 0
        self.prev_bias_gradient = self.lr
        self.prev_bias_step = self.lr

        prev_loss = None
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

class SupportVectorMachineModel(Model):
    """
    SupportVectorMachineModel is a wrapper for sklearn's SVM

    Attributes
    ----------
    model : sklearn.svm.SVC
        The model to train and make predictions with.
    """
    def __init__(self):
        self.model = svm.SVC()

    def __str__(self) -> str:
        return "SupportVectorMachineModel"

    @timeit
    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)
    
    def accuracy(self, x, y_true):
        y_pred = self.predict(x)

        return accuracy_score(y_true, y_pred)
        
    def loss(self, x, y_true):
        y_pred = self.model.decision_function(x)

        return hinge_loss(y_true, y_pred)
    
class DeepLearningModel(Model):
    """
    DeepLearningModel is an implementation of a basic feedforward neural network using Keras

    Parameters
    ----------
    tol : float
        The tolerance at which to stop iterating.
    
    Attributes
    ----------
    model : tensorflow.keras.Model
        The model to train and make predictions with.
    """
    def __init__(self, tol=1e-4):
        self.tol = tol
        self.model = Sequential()
        self.model.add(Dense(12, input_shape=(65,), activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def __str__(self) -> str:
        return "DeepLearningModel"
    
    @timeit
    def train(self, x_train, y_train, epochs=300, batch_size=64, verbose=0):
        callbacks = [EarlyStopping(monitor='loss', min_delta=self.tol)]
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)

    def predict(self, x):
        return (self.model.predict(x) > 0.5).astype(int)

    def accuracy(self, x, y_true):
        _, accuracy = self.model.evaluate(x, y_true, verbose=0)

        return accuracy

    def loss(self, x, y_true):
        loss, _ = self.model.evaluate(x, y_true, verbose=0)

        return loss