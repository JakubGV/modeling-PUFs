import time
import argparse
import pandas as pd
import numpy as np

from numpy import array

from sklearn.metrics import accuracy_score

from PUFmodels import XORArbPUF
from predictor import MCError, LRError, TrainData, prodLinearPredictor,\
                      BasicTrainable, GradLearner, RProp, Closures, customPredictor

def xorKnacker(bit_count, numxor, accuracy, accuracy2, ArbPUFgoal, how_often, CRP):
    sucess = 0
    oft = 0
    mc_rate = MCError()
    
    while sucess < how_often:
        erf = LRError()
        features = ArbPUFgoal.calc_features(ArbPUFgoal.generate_challenge(CRP))
        set = TrainData(features, ArbPUFgoal.bin_response(features))
        
        testfeatures = ArbPUFgoal.calc_features(ArbPUFgoal.generate_challenge(10000))
        testtargets = ArbPUFgoal.bin_response(testfeatures)
              
        performanceTrain = 1
        count = 0
        start = time.time()
        
        while performanceTrain > accuracy:
            count += 1
            
            model = prodLinearPredictor(bit_count + 1, numxor)

            lesson = BasicTrainable(set, model, erf)
            learner = GradLearner(lesson, RProp([bit_count + 1] * numxor), Closures(accuracy=accuracy2).grad_performance_stop)
            learner.evaluate_lesson()
            
            performanceTrain = mc_rate.calc(lesson.trainset.targets, lesson.response()) / set.targets.size
            
            print ('Train:', performanceTrain)
            
        responses = model.response(testfeatures)
        performanceTest = mc_rate.calc(testtargets, responses) / testtargets.shape[0]
        print ('Test:', performanceTest, 'time since start:', time.time() - start)

        grounded_responses = [1 if r > 0 else -1 for r in responses]
        print ('Test accuracy:', accuracy_score(testtargets, grounded_responses), 'time since start:')

        sucess += 1
        oft += 1
             
    print('Finished')

def xorKnackertester(bit_count, numxor, accuracy, accuracy2, how_often, CRParray):
    ArbPUFgoal = XORArbPUF(bit_count, numxor, 'equal')

    for i in range(CRParray.size):
        xorKnacker(bit_count, numxor, accuracy, accuracy2, ArbPUFgoal, how_often, int(CRParray[i]))

def run_on_my_data():
    # Get my data to use
    
    # Read the challenge data
    challenges = pd.read_csv('../data/APUF_XOR_Challenge_Parity_64_500000.csv', header=None, engine='pyarrow')
    challenges = challenges.iloc[:, :65] #grab the 65 bits of the challenge

    # Read the response data
    xor_2 = pd.read_csv('../data/2-xorapuf.csv', header=None)

    # Select NUM_POINTS points
    NUM_POINTS = 20000
    x = challenges.iloc[:NUM_POINTS].to_numpy()
    x = np.transpose(x) #put it into the (65 bit challenge rows, 20000 sample columns form)
    y = np.ndarray.flatten(xor_2.iloc[:NUM_POINTS].to_numpy())
    
    # Initialize other variables needed
    mc_rate = MCError()
    erf = LRError()
    accuracy2 = 0.01
    bit_count = 64

    features = x
    set = TrainData(features, y)

    # Train
    model = customPredictor(bit_count + 1)

    lesson = BasicTrainable(set, model, erf)
    learner = GradLearner(lesson, RProp([bit_count + 1]), Closures(accuracy=accuracy2).grad_performance_stop)
    learner.evaluate_lesson()
    
    performanceTrain = mc_rate.calc(lesson.trainset.targets, lesson.response()) / set.targets.size
    print ('Train:', performanceTrain, 'time since start:')
    
    # Test
    responses = model.response(x)
    grounded_responses = [1 if r > 0 else 0 for r in responses]
    performanceTest = mc_rate.calc(y, responses) / y.shape[0]
    print(f"Test error: {performanceTest}")
    accuracy = accuracy_score(y, grounded_responses)
    print(f"Test accuracy: {accuracy:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description="Run the paper's model on their generated data or the new data."
    )
    parser.add_argument('-n', '--new-data', action='store_true', help="run with the new data")
    args = parser.parse_args()
    
    if args.new_data:
       run_on_my_data() 
    else:
        xorKnackertester(64, 2, 0.05, 0.01, 1, array([20000]))