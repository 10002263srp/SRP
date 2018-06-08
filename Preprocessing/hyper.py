# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 11:38:39 2018

@author: Daniel
"""

from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice

from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def data():
    '''
    Data providing function:
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    data = pd.read_csv("C:/Users/Daniel/Desktop/code/bcancer/bcancerdata.csv",header = 0)
    data.drop("Unnamed: 32",axis=1,inplace=True)
    data.drop("id", axis=1, inplace=True)
    
    prediction_var = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
    X = data[prediction_var].values
    Y = data.diagnosis.values
    
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X[:285], encoded_Y[:285], test_size=0.2, random_state=2018) 
    return X_train, y_train, X_test, y_test

def model(X_train, y_train, X_test, y_test):
    '''
    Model providing function:
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    model.add(Dense({{choice([16, 32, 64, 128, 256])}}, 
                      input_dim=30, 
                      activation={{choice(['sigmoid', 'softmax','selu','tanh','linear','relu'])}}))
   
    #Hidden layer
    model.add(Dense({{choice([16, 32, 64 , 128, 256])}}))
    model.add(Activation('relu'))
    
    #model.add(Dense({{choice([16, 32, 64 , 128, 256])}}))
    #model.add(Activation('relu'))

    #Output layer
    model.add(Dense(1, activation={{choice(['sigmoid', 'softmax','selu','tanh','linear', 'relu'])}}))
    
    #compile model
    model.compile(loss={{choice(['binary_crossentropy', 'mse', 'logcosh', 'kullback_leibler_divergence', 'poisson', 'cosine_proximity'])}}, 
                  optimizer={{choice(['rmsprop', 'adam', 'sgd', 'nadam'])}}, 
                  metrics=['accuracy'])


    model.fit(X_train, y_train,
              batch_size={{choice([16, 32, 64, 128])}},
              epochs=2,
              verbose=2,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    X_train, y_train, X_test, y_test = data()

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials(),
                                          )
    
    print("Best performing model: ", best_run)
    print("Evalutation of best performing model:", best_model.evaluate(X_test, y_test))
    