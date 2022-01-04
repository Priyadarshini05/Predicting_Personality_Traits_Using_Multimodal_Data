import sys
from argparse import ArgumentParser
import os
import warnings
from config import conf 
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import xgboost as xgb
import getopt
import threading
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import random
from sklearn.model_selection import RandomizedSearchCV

warnings.filterwarnings("ignore")


def tuning(clf, param_name, tuning_params, data, labels, kf):
    labels = labels
    data= data
    acc_history = []
    
    for param in tqdm(tuning_params):
        # initialize history list
        train_accuracy_history = []
        val_accuracy_history = []
        
        # setup classifier
        xgb_clf = clf['a']
        xgb_clf.set_params(**{param_name: param})
        for train_idx, val_idx in kf.split(data):
            
            # normalize using mean and std
            data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
            # collect data for cross validation
            train_data, val_data = data[train_idx], data[val_idx]
            train_labels, val_labels = labels[train_idx], labels[val_idx]
            # fit classifier
            xgb_clf.fit(train_data, train_labels)
            # predict arousal and valence
            train_predict_labels = xgb_clf.predict(train_data)
            val_predict_labels = xgb_clf.predict(val_data)
            # metrics calculation
            train_accuracy = f1_score(train_labels, train_predict_labels, average='macro')
            val_accuracy = f1_score(val_labels, val_predict_labels, average='macro')
            train_accuracy_history.append(train_accuracy)
            val_accuracy_history.append(val_accuracy)
            
        train_mean_accuracy = np.mean(train_accuracy_history)
        val_mean_accuracy = np.mean(val_accuracy_history)
        acc_history.append(val_mean_accuracy)
       
    print('Tuning Result:')
    max_idx = np.argmax(acc_history)
    print("Tuned best parametrs: Best value = {:.2f}, acc = {:.4f}".format(
        tuning_params[max_idx], acc_history[max_idx]))
    return tuning_params[max_idx], acc_history[max_idx]

def main():
    """ Main function
    """

    amigos_data = np.loadtxt('features_all_20s.csv',skiprows=1, delimiter=',')
    labels = np.loadtxt('Final_Personality_20s.csv',skiprows=1, delimiter=',')
    ids=np.loadtxt('ids_20s.csv',skiprows=1, delimiter=',')
    labels=labels[:, 1]
    kf = KFold(n_splits=2)
    gkf=GroupKFold(n_splits=5)

    # tune XGB classifier parameters
    grid_search_params_xgb = {
        'max_depth': [3,4,5],
        'n_estimators': [10,15,20]
    }
    other_tuning_params_xgb = {
        'learning_rate': np.arange(0.01, 0.41, 0.01),
        'gamma': np.arange(0, 10.1, 0.5),
        'min_child_weight': np.arange(0.80, 1.21, 0.01),
        'max_delta_step': np.arange(0, 2.05, 0.05),
        'subsample': np.arange(1.00, 0.59, -0.01),
        'colsample_bytree': np.arange(1.00, 0.09, -0.01),
        'colsample_bylevel': np.arange(1.00, 0.09, -0.01),
        'reg_alpha': np.arange(0, 2.05, 0.05),
        'reg_lambda': np.arange(0.50, 2.55, 0.05),
        'scale_pos_weight': np.arange(0.80, 1.21, 0.01),
        'base_score': np.arange(0.40, 0.61, 0.01),
        'seed': np.arange(0, 41)
    }

    
    # XGB grid search tuning
    best_params = {
        'max_depth': 3,
        'n_estimators': 20
    }
    acc = 0
    print('Tuning max_depth and n_estimators')
    for param in grid_search_params_xgb['max_depth']:
        print('in grid search')
        print('max_depth', param)
        xgb_clf = {
            'a': xgb.XGBClassifier(max_depth=param, objective="binary:logistic"),
            }
        tuning_params = grid_search_params_xgb['n_estimators']
        param, tmp_acc = tuning(
            xgb_clf, 'n_estimators', tuning_params, amigos_data, labels, kf)
        print('param',param,'tmp_acc',tmp_acc)
        if tmp_acc >= acc:
            best_params['max_depth'] = param
            best_params['n_estimators'] = param
            acc = tmp_acc
    # XGB tune other parameters
    for param_name, tuning_params in other_tuning_params_xgb.items():
        print('Tuning', param_name)
        xgb_clf = {
            'a': xgb.XGBClassifier(objective="binary:logistic"),
              }
        xgb_clf['a'].set_params(**best_params)
     
        param,_ = tuning(
            xgb_clf, param_name, tuning_params, amigos_data, labels, kf)
        best_params[param_name] = param
        


    # tune RF parameters
    grid_search_params_rf = {
               'max_features': [10,15,20],
               'max_depth': [3,5,10],
               }
    
    rf_clf=RandomForestClassifier()   
    rf_random = RandomizedSearchCV(estimator = rf_clf, param_distributions = grid_search_params_rf, n_iter = 100, cv = gkf, verbose=2, random_state=42, n_jobs = 5)
    rf_random.fit(amigos_data,np.ravel(labels),groups=ids)

    #the optimized hyperparameters
    print('XGBoost best parameters:', best_params) 
    print('Random forest best parameters:',rf_random.best_params_)


if __name__ == '__main__':


    main()