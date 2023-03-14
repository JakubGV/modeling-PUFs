from scipy import vstack, newaxis, arange, sign, dot, array, prod, \
                  ones, zeros, concatenate, log, swapaxes, empty, tanh, \
                  hstack, linspace, around, mean, std, nonzero, isinf
#added
from numpy import random
import numpy as np
#added
from scipy import exp as numexp
import csv
from copy import deepcopy
import random as rd

class customPredictor:
    '''linearPredictor provides methods to learn and evaluate a linear model
        based on a weight vector. Methods operate on 2D Array features, with
        the 0th dimension corresponding to different features and the first
        dimension to different samples.
        
        attributes:
        parameter -- the weight vector
        
        methods:
        response(features) -- calculates the result of the predictor to
            features with respect to the current parameter
        grad(features) -- gives the gradient of the linear model
        shift_param(step) -- changes parameter of instance by step
    '''
    
    def __init__(self, dim, mean=0, stdev=1):
        '''creates a linearPredictor with normal distributed weights
        
            Keyword Arguments:
            dim -- dimension of the weight vector
            mean(optional) -- mean of the weight distribution
            stdev(optional) -- standard deviation of the weight distribution
        '''
        
        self.dim = dim
        self.parameter = np.zeros(dim)

    def response(self, features):
        ''' gives response of the predictor to the features
            
            Keyword Arguments:
            features -- 2D arrays of feature vectors; 0th dim features,
                1st dim samples
            
            returns: 1D array with response for all feature samples        
        '''
        response = dot(self.parameter, features)
        return response

    def grad(self, error_fct_derivativ, features):
        '''gives gradient of the linear predictor with respect to its parameter
           based on the error function derivativ with respect to the 
           predictor function 
        
            Keyword Arguments:
            features -- 2D array of feature vectors; 0th dimension features,
                1st dimension samples
            error_fct_derivativ -- 1D array; derivativ of the error function
                with respect to the predictor function (at current vaulue);
                dimension is sample size
                
            returns: single element list with 1D array (dimension equiv. to
                self.parameter) as the batch gradient obtained by chainrule 
                of error_fct_derivativ and gradient of the linear predictor 
        '''
        return [dot(error_fct_derivativ, features.transpose())]
    
    def shift_param(self, step):
        ''' change parameter by amount of step
        
        Keyword Arguments:
        step -- single element list of 1D array wth dimension as self.parameter
        
        Side Effects:
        changes the instance variable parameter
        
        Exeptions:
        DimensionError -- dimension of step and self.parameter do not match
        '''
        step = step[0]
        if step.shape != self.parameter.shape:
            raise Exception('Dimension error')
        else: self.parameter += step

class linearPredictor(object):
    
    '''linearPredictor provides methods to learn and evaluate a linear model
        based on a weight vector. Methods operate on 2D Array features, with
        the 0th dimension corresponding to different features and the first
        dimension to different samples.
        
        attributes:
        parameter -- the weight vector
        
        methods:
        response(features) -- calculates the result of the predictor to
            features with respect to the current parameter
        grad(features) -- gives the gradient of the linear model
        shift_param(step) -- changes parameter of instance by step
    '''
    
    def __init__(self, dim, mean=0, stdev=1):
        '''creates a linearPredictor with normal distributed weights
        
            Keyword Arguments:
            dim -- dimension of the weight vector
            mean(optional) -- mean of the weight distribution
            stdev(optional) -- standard deviation of the weight distribution
        '''
        
        self.dim = dim
        self.parameter = random.normal(mean, stdev, dim)

    def response(self, features):
        ''' gives response of the predictor to the features
            
            Keyword Arguments:
            features -- 2D arrays of feature vectors; 0th dim features,
                1st dim samples
            
            returns: 1D array with response for all feature samples        
        '''
        response = dot(self.parameter, features)
        return response

    def grad(self, error_fct_derivativ, features):
        '''gives gradient of the linear predictor with respect to its parameter
           based on the error function derivativ with respect to the 
           predictor function 
        
            Keyword Arguments:
            features -- 2D array of feature vectors; 0th dimension features,
                1st dimension samples
            error_fct_derivativ -- 1D array; derivativ of the error function
                with respect to the predictor function (at current vaulue);
                dimension is sample size
                
            returns: single element list with 1D array (dimension equiv. to
                self.parameter) as the batch gradient obtained by chainrule 
                of error_fct_derivativ and gradient of the linear predictor 
        '''
        return [dot(error_fct_derivativ, features.transpose())]
    
    def shift_param(self, step):
        ''' change parameter by amount of step
        
        Keyword Arguments:
        step -- single element list of 1D array wth dimension as self.parameter
        
        Side Effects:
        changes the instance variable parameter
        
        Exeptions:
        DimensionError -- dimension of step and self.parameter do not match
        '''
        step = step[0]
        if step.shape != self.parameter.shape:
            raise Exception('Dimension error')
        else: self.parameter += step

class prodLinearPredictor(object):
    def __init__(self, dim, num_prod, mean=0, stdev=1):
        self.dim = dim
        self.num_prod = num_prod
        # create indivudual ArbiterPUFs
        self.indiv_linpredictor = [linearPredictor(dim, mean, stdev) for i in range(num_prod)]
        self._indiv_response = []
        self._response = []
        
    def response(self, features):
        self._indiv_response = empty([self.num_prod, features.shape[-1]])
        for predictor in range(self.num_prod):
            self._indiv_response[predictor, :] = dot(self.indiv_linpredictor[predictor].parameter, features[predictor, :, :])
        self._response = prod(self._indiv_response, 0)
        return self._response
      
    def grad(self, error_fct_derivativ, features):
        grad = list(range(self.num_prod))
        for i in grad:
            grad[i] = dot(error_fct_derivativ / self._indiv_response[i, :] * self._response, swapaxes(features[i, :, :], 0, 1))
        return grad
    
    def shift_param(self, step):
        for predictor, indiv_step in enumerate(step):
            self.indiv_linpredictor[predictor].shift_param([indiv_step])

class Sigmoid(object):
    def __init__(self, scale_factor=1):
        self.scale = scale_factor
    
    def calc(self, input_array):
        out = 1. / (1 + numexp(-self.scale * input_array))
        return out
        
    def grad(self, input_array):
        out = input_array * (1. - input_array)
        return out
            
class LRError(object):
    def __init__(self, scale=1):
        self.sigmoid = Sigmoid(scale)
        
    def calc(self, targets, response):
        errors = (log(1 + numexp(-self.sigmoid.scale * targets.squeeze() * response.squeeze())))
        errors[isinf(errors)] = 1000
        return sum(errors)
    
    def grad(self, targets, response):
        class_prob = self.response_interpretation(response)
        error_fct_derivative = class_prob - (targets + 1) / 2
        return error_fct_derivative
    
    def response_interpretation(self, response):
        return self.sigmoid.calc(response)

class MSError(object):
    def calc(self, targets, response):
        error = sum((targets - response.squeeze()) ** 2) / targets.shape[0]
        return error
    
    def grad(self, targets, response):
        return response - targets
    
class MCError(object):
    
    def calc(self, targets, response):
        error = sum(1 - targets.squeeze() * sign(response.squeeze())) / 2 
        return error
                         
class RProp(object):
    def __init__(self, dimension, initial_stepsize=1, etaminus=0.5, etaplus=1.2):
        self.etaminus = etaminus
        self.etaplus = etaplus
        self.gradold = [ones(dim) for dim in dimension]
        self.stepold = [zeros(dim) for dim in dimension]
        self.stepsize = [ones(dim) * initial_stepsize for dim in dimension]
        self.step = [zeros(dim) for dim in dimension]
        
    def update_step(self, grad):
        
        for part_num, grad_part in enumerate(grad):
            stepindicator = sign(grad_part * self.gradold[part_num])
        
            self.stepsize[part_num][stepindicator > 0] *= self.etaplus
            self.stepsize[part_num][stepindicator < 0] *= self.etaminus

            self.step[part_num][stepindicator > 0] = -(self.stepsize[part_num][stepindicator > 0] * sign(grad_part[stepindicator > 0]))
            self.step[part_num][stepindicator < 0] = -self.stepold[part_num][stepindicator < 0]
            self.step[part_num][stepindicator == 0] = -self.stepsize[part_num][stepindicator == 0] * sign(grad_part[stepindicator == 0])

            self.gradold[part_num] = grad_part
            self.gradold[part_num][stepindicator < 0] = 0
            self.stepold[part_num] = self.step[part_num]
        
        return self.step  

class GradientDescent(object):
    def __init__(self, learnrate=1):
        self.learnrate = learnrate
        
    def update_step(self, grad):
        step = []
        for component in grad:
            step.append(-component * self.learnrate)
        return step

class AnealingGradientDescent(object):
    def __init__(self, learnrate=1, decay=0.999):
        self.learnrate = learnrate
        self.decay = decay
        
    def update_step(self, grad):
        self.learnrate *= self.decay
        step = []
        for component in grad:
            step.append(-component * self.learnrate)
        return step
        
class Trainable(object):
        
    def current_error(self):
        pass     
    
    def update(self, step):
        pass
        
    def evaluate_lesson(self):
        pass
          
class BasicTrainable(Trainable):
    def __init__(self, trainset, model, errorfct):
        self.trainset = trainset
        self.model = model
        self.errorfct = errorfct
        
    def response(self, param={}):
        features = param['features'] if ('features' in param) else self.trainset.features
        return self.model.response(features)
        
    def current_error(self, param={}):
        targets = param['targets'] if ('targets' in param) else self.trainset.targets
        response = self.response(param) if ('features' in param) else self.response()
        return self.errorfct.calc(targets, response)        
    
    def update(self, step):
        self.model.shift_param(step)
        #self.response = self.model.response(self.trainset.features)
        
    def evaluate_lesson(self):
        return self.current_error
    
    def response_interpretation(self, param={}):
        features = param['features'] if ('features' in param) else self.trainset.features
        response = param['response'] if ('response' in param) else self.response({'features':features})
        return self.errorfct.response_interpretation(response)
    
class Learner(Trainable):
    def __init__(self, closure_fct, trainable):
        self.lesson = trainable
        self.closure_fct = closure_fct 
    
    def gettrainset(self):
        return self.lesson.trainset
    
    def settrainset(self, trainset):
        self.lesson.trainset = trainset 
    
    def response(self, param={}):
        return self.lesson.response(param)        
    
    def current_error(self, param={}):
        return self.lesson.current_error(param)

    def update(self, step):
        self.lesson.update(step)
    
    def response_interpretation(self, param={}):
        return self.lesson.response_interpretation(param)
    
    def evaluate_lesson(self):
        pass
    
    
        
class GradLearner(Learner):        
    def __init__(self, lesson, grad_learning_strategy, closure_fct):
        self.gradstrat = grad_learning_strategy
        Learner.__init__(self, closure_fct, lesson)
        
    def evaluate_lesson(self):
        grad = []
        while self.closure_fct(self.lesson, grad):
            grad = self.grad()
            self.update(self.gradstrat.update_step(grad))
        return self.current_error()
        
    def grad(self):
        return self.lesson.model.grad(self.lesson.errorfct.grad(self.lesson.trainset.targets, self.lesson.response()),
                                      self.lesson.trainset.features)
            
class Closures(object):
    def __init__(self, stop_iteration=1E5000, accuracy=0.001):
        self.mc_error = MCError()
        self.ms_error = MSError()
        self.iteration_count = 0
        self.stop_grad = 0.0001
        self.accuracy = accuracy
        self.stop_iteration = stop_iteration
        self.error = 1E5000
            
    def reset(self):
        self.iteration_count = 0
        self.error = 1E5000
    
    def __call__(self, lesson, grad):
        return self.grad_performance_stop(lesson, grad)
        
    def mc_zero(self, lesson, grad):
        error = self.mc_error.calc(lesson.trainset.targets, lesson.response) 
        print (error / lesson.trainset.targets.shape[0])
        return  error != 0
    
    def num_iterations(self, lesson, grad):
        self.iteration_count += 1
        if self.iteration_count % 50 == 1:
            self.error = self.mc_error.calc(lesson.trainset.targets, lesson.response())
            #lesson.model.draw(self.iteration_count)
            print (self.iteration_count, self.error)
        return self.iteration_count != self.stop_iteration + 1 
    
    def grad_performance_stop(self, lesson, grad):
     
        self.iteration_count += 1
        if grad:    
            abs_grad = [abs(grad_part) for grad_part in grad]
            total_grad = sum(sum(abs_grad).flatten())
        else:
            total_grad = self.stop_grad + 1    
        train_performance = self.mc_error.calc(lesson.trainset.targets,
                                               lesson.response()
                                               ) / lesson.trainset.targets.shape[0]
        
        if self.iteration_count % 500 == 1:    
            print (self.iteration_count, total_grad, train_performance)
        
        return ((total_grad > self.stop_grad) 
                and (train_performance > self.accuracy) 
                and (self.iteration_count < self.stop_iteration))
            
class TrainData(object):
    ''' TrainData acts as a container for traindata and will give methods for 
        reading in data
        
        attributes:
        features -- array of features, where last dimension runs over samples
        targets -- 1D array of results for all samples
    '''
    def __init__(self, features=empty(0), targets=empty(0)):
        self._features = features
        self.targets = targets
        self.scaling = (0, 1)
        self.offset_features = False
        self.samplesize = targets.shape[0] if features.any() else 0
    
    @property
    def features(self):
        return self._features
    
    def scale_self(self):
        data_mean = mean(self._features, axis= -1)
        data_std = std(self._features, axis= -1)
        self.scaling = (data_mean, data_std)  
        self._features = self.scale_same(self._features)
        
    def scale_same(self, features):
        scaled_features = empty(features.shape)
        for ind in range(features.shape[-1]):
            scaled_features[:, ind] = (features[:, ind] - self.scaling[0]) / self.scaling[1]
        if self.offset_features:
            scaled_features = concatenate((scaled_features, ones((1, scaled_features.shape[-1]))))
        return scaled_features
    
    def add_offsetfeature(self):
        self.offset_features = True
        self._features = concatenate((self._features, ones((1, self.samplesize))))
    
    def load(self, file):
        csvreader = csv.reader(open(file), delimiter=' ')
        filecontent = array([i for i in csvreader], dtype='d')
        self.targets = array(filecontent[:, 0])
        self._features = array(filecontent[:, 1:].transpose())
        
    def feature_subset(self, index):
        return TrainData(self.features[index], self.targets)
    
    def sample_subset(self, index):
        return TrainData((self.features.transpose()[index]).transpose(),
                                                        self.targets[index])
    
    def group_same_samples(self, indices=[]):
        indices = indices if indices else range(self._features.shape[-1])
        sample_groups = []
        while indices:
            existed = False
            sample = indices.pop()
            for group in sample_groups:
                if all(self._features[:, group[0]] == self._features[:, sample]):
                    group.append(sample)
                    existed = True
                    break
            if not(existed):
                sample_groups.append([sample])
        mean_value = [mean([self.targets[ind] for ind in group]) for group in sample_groups]    
        return (sample_groups, mean_value)         

    def stratificate(self, bin_borders, characteristic, items):
        ''' returns list of lists, each list containing items with characteristic
            between bin_borders
        '''
        bins = []
        characteristic = array(characteristic)
        low_border = bin_borders[0]
        for up_border in bin_borders[1:]:
            bins.append(list(nonzero((characteristic >= low_border) & 
                                             (characteristic < up_border))[0]))
            low_border = up_border
        bins[-1] += list(nonzero(characteristic == up_border)[0])
        
        for strat_group in bins:
            for list_pos, item_num in enumerate(strat_group):
                strat_group[list_pos] = items[item_num]
        
        return bins