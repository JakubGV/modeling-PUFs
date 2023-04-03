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
    response_names = ['apuf', '2-xorapuf', '3-xorapuf', '4-xorapuf', '5-xorapuf']
    responses = {}
    for name in response_names:
        path = './data/' + name + '.csv'
        r = pd.read_csv(path, header=None, engine='pyarrow')
        r = np.ndarray.flatten(r.to_numpy()) #flatten into a 1D array
        responses[name] = r

    # Different number of points to try
    NUM_POINTS = (5000, 10000, 20000, 50000, 100000, 500000)

    # Store results
    results = {
        'Name': [],
        'CRP': [],
        'Model': [],
        'Accuracy': [],
        'Time': [],
    }
    
    for name in response_names:
        print(
            "--------------------\n"
            f"{name}\n"
            "--------------------\n"
        )
        r = responses[name]
    
        for num_points in NUM_POINTS:
            print(
                "--------------------"
                f"{num_points:,} points"
                "--------------------"
            )
            x = challenges[:num_points]
            y = r[:num_points]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
            
            # Models to run
            models = [LogisticRegressionModel(), RPropLogisticRegressionModel(), SupportVectorMachineModel(), DeepLearningModel()]
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

                results['Name'].append(name)
                results['CRP'].append(num_points)
                results['Model'].append(str(model))
                results['Accuracy'].append(accuracy)
                results['Time'].append(training_time)

            pd.DataFrame(results).to_csv('results.csv', index=False)