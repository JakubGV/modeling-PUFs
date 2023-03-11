import time

from sklearn import svm
from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self) -> None:
        self.model = LogisticRegression(solver='newton-cholesky', random_state=0)

    def train(self, x, y) -> None:
        start = time.time()
        self.model.fit(x, y)
        print(f"Training took: {time.time()-start:.2f}s")

    def predict(self, x):
        return self.model.predict(x)
    
    def score(self, x, y):
        score = self.model.score(x, y)
        print(f"Accuracy: {score:.2%}")
        
        return score
    
class SVMModel:
    def __init__(self) -> None:
        self.model = svm.SVC()

    def train(self, x, y) -> None:
        start = time.time()
        self.model.fit(x, y)
        print(f"Training took: {time.time()-start:.2f}s")

    def predict(self, x):
        return self.model.predict(x)
    
    def score(self, x, y):
        score = self.model.score(x, y)
        print(f"Accuracy: {score:.2%}")
        
        return score