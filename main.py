import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from models import LogisticRegressionModel, RPropLogisticRegressionModel, SupportVectorMachineModel, DeepLearningModel

if __name__ == '__main__':
    # Read the challenge data
    challenges = pd.read_csv('./data/APUF_XOR_Challenge_Parity_64_500000.csv', header=None, engine='pyarrow')
    challenges = challenges.iloc[:, :65] #grab the 65 bits of the challenge input
    challenges = challenges.to_numpy() #2D array of shape (#challenges, #bits)

    # Read the response data
    xor_2_responses = pd.read_csv('./data/2-xorapuf.csv', header=None, engine='pyarrow')
    xor_2_responses = np.ndarray.flatten(xor_2_responses.to_numpy()) #flatten into a 1D array

    # Different number of points to try
    NUM_POINTS = (5000, 10000, 20000, 50000, 100000, 500000)

    # Models to run
    models = [LogisticRegressionModel(), RPropLogisticRegressionModel(), SupportVectorMachineModel(), DeepLearningModel()]

    for num_points in NUM_POINTS:
        x = challenges[:num_points]
        y = xor_2_responses[:num_points]
        print(
            "--------------------"
            f"{num_points} points"
            "--------------------"
        )

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        for model in models:
            training_time = model.train(x_train, y_train)
            loss = model.loss(x_train, y_train)
            accuracy = model.accuracy(x_test, y_test)
            print(f"{model}\n"
                  f"----------\n"
                  f"Training took {training_time:.3f}s\n"
                  f"Final training loss: {loss:.3f}\n"
                  f"Testing accuracy: {accuracy:.2%}\n"
            )