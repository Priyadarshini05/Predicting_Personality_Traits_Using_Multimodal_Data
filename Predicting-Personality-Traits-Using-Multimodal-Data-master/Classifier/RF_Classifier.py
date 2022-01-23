import sys
import numpy as np
import conf 
import os
import getopt
import threading
from sklearn.model_selection import  KFold as LKF
from sklearn.model_selection import  StratifiedKFold as SKF
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
import random
from matplotlib import pyplot
import glob


def predict_all():
    # add threads to a list, and wait for all of them in the end
    threads = []
    
    for trait in range(0,5):
        print('trait',trait)
        for si in range(low_repetitions, num_repetitions):
            thread = threading.Thread(target=save_predictions, args=(trait, conf.get_result_filename(trait, shuffle_labels, si), si))
            sys.stdout.flush()
            thread.start()
            threads.append(thread)  
     
    for thread in threads:
        thread.join()
        print ('waiting to join')
        
def load_data(t, chosen_features = None):
     
    x= np.genfromtxt(os.path.join('Data/Features_All_20s.csv'),skip_header=1,delimiter=',')
    y=np.genfromtxt('Data/Final_Personality_20s.csv',skip_header=1,delimiter=',').astype(int)[:,t]
    ids=np.genfromtxt('Data/ids_20s.csv',skip_header=1,delimiter=',')
    print('loaded data for trait:',t+1)

    # load the features chosen in the inner validation  
    if chosen_features is not None:
        x = x[:,chosen_features]
    return x, y, ids


def save_predictions(t, filename, rs):
           
    # create RandomForest classifier with parameters given in _conf.py
    clf = RandomForestClassifier(random_state=rs, verbose=verbosity, class_weight='balanced',
	                            n_estimators=conf.n_estimators, n_jobs=conf.max_n_jobs, max_features=conf.tree_max_features,
	                            max_depth=conf.tree_max_depth)
       
    # use ground truth to create folds for outer cross validation in a stratified way, i.e. such that
    # each label occurs equally often
    
    participant_scores = np.genfromtxt('Data/Binned_Personality.csv',skip_header=1,delimiter=',').astype(int)[:,t+1]
    outer_cv=SKF(conf.n_outer_folds,shuffle=True,random_state=True)
    len_outer_cv=outer_cv.get_n_splits(participant_scores)
    
    # initialise arrays to save information
    feat_imp = np.zeros((len_outer_cv, conf.max_n_feat))  # feature importance
    preds = np.zeros((conf.n_participants), dtype=int)  # predictions on participant level
    x= np.zeros(31) # placeholder for X instead of actual training data
    for outer_i, (outer_train_participants, outer_test_participants) in enumerate(outer_cv.split(x, participant_scores)):
        print (str(outer_i + 1) + '/' + str(conf.n_outer_folds))
        
        # find best window size in inner cv, and discard unimportant features
        inner_performance = np.zeros((conf.n_inner_folds, 1))
        inner_feat_importances = np.zeros((conf.max_n_feat, 1))
   
        #load all the extracted features
        x_all, y_all, ids_all = load_data(t)
        if shuffle_labels:
            np.random.seed(316588 + 111 * t + rs)
            perm = np.random.permutation(len(y_all))
            y_all = y_all[perm]
            ids_all = ids_all[perm]
      
        # cut out the outer train samples
        outer_train_samples = np.array([p in outer_train_participants for p in ids_all])
        outer_train_x = x_all[outer_train_samples, :]
        outer_train_y = y_all[outer_train_samples]
        outer_train_y_ids = ids_all[outer_train_samples]

        # build inner cross validation such that all samples of one person are either in training or testing
        inner_cv=LKF(n_splits=conf.n_inner_folds)
        for inner_i, (inner_train_indices, inner_test_indices) in enumerate(inner_cv.split(outer_train_y_ids)):
            
        # create inner train and test samples. Note: both are taken from outer train samples!
            inner_x_train = outer_train_x[inner_train_indices, :]
            inner_y_train = outer_train_y[inner_train_indices]
            inner_x_test = outer_train_x[inner_test_indices, :]
            inner_y_test = outer_train_y[inner_test_indices]
           
        # fit Random Forest
            clf.fit(inner_x_train, np.ravel(inner_y_train))
           
        # save predictions and feature importance
            inner_pred = clf.predict(inner_x_test)
            inner_pred=inner_pred.reshape(-1,1)
            inner_feat_importances[:, 0] += clf.feature_importances_

        # compute and save performance in terms of accuracy
            innerpreds = []
            innertruth = []
            inner_test_ids = outer_train_y_ids[inner_test_indices]
            for testp in np.unique(inner_test_ids):
                (values, counts) = np.unique(inner_pred[inner_test_ids == testp], return_counts=True)
                ind = np.argmax(counts)
                innerpreds.append(values[ind])
                innertruth.append(inner_y_test[inner_test_ids == testp][0])
            inner_performance[inner_i, 0] = accuracy_score(np.array(innertruth), np.array(innerpreds))
            print ('ACC: ', '%.2f' % (inner_performance[inner_i, 0] * 100))

        # evaluate classifier on outer cv using the most informative features
        chosen_i = np.argmax(np.mean(inner_performance, axis=0))
        chosen_features = (inner_feat_importances[:,chosen_i]/float(conf.n_inner_folds)) > 0.005

        # reload all data
        x, y, ids = load_data(t, chosen_features=chosen_features)
        if shuffle_labels:
            np.random.seed(316588 + 111 * t + rs + 435786)
            perm = np.random.permutation(len(y))
            y = y[perm]
            ids = ids[perm]
        outer_train_samples = np.array([p in outer_train_participants for p in ids])
        outer_test_samples = np.array([p in outer_test_participants for p in ids])
        if outer_train_samples.size > 0 and outer_test_samples.size > 0:
            x_train = x[outer_train_samples, :]
            y_train = y[outer_train_samples]
            x_test = x[outer_test_samples, :]
            y_test = y[outer_test_samples]

            # fit Random Forest
            clf.fit(x_train, np.ravel(y_train))
            pred = clf.predict(x_test)
            pred=pred.reshape(-1,1)            
            for testp in outer_test_participants:               
                if testp in ids[outer_test_samples]:
                    # majority voting over all samples that belong to participant testp
                    (values, counts) = np.unique(pred[ids[outer_test_samples] == testp], return_counts=True)
                    ind = np.argmax(counts)
                    preds[testp] = values[ind]                    
                else:
                    # participant does not occour in outer test set
                    preds[testp] = -1
            # save the resulting feature importance
            feat_imp[outer_i, chosen_features] = clf.feature_importances_
        else:
            for testp in outer_test_participants:
                preds[testp] = -1
        feat_imp[outer_i, chosen_features] = -1
            
    # compute resulting F1 score and save to file
    nonzero_preds = preds[preds>-1]
    nonzero_truth = participant_scores[preds>-1]
    f1 = f1_score(nonzero_truth, nonzero_preds, average='macro')
    accuracy=accuracy_score(nonzero_truth,nonzero_preds)
    np.savez(filename, f1=f1, accuracy=accuracy, feature_importances=feat_imp,inner_feat_importances='inner_feat_importances')

print('Success')    
#print the f1_score stored in the result file
def F1_score():
    for i in range(0,len(conf.personality)):
        sum=0
        for name in glob.glob('Classifier/trait_'+str(i+1)+'.npz'):
            a=np.load(name)
            b=a['f1']
            sum=sum+b
        Mean=sum/1
        print(conf.personality[i],"F1_score:",Mean)  

#print the accuracy stored in the result file
def accuracy(): 
    for i in range(0,len(conf.personality)):
        sum=0
        for name in glob.glob('Classifier/trait_'+str(i+1)+'.npz'):
            a=np.load(name)
            b=a['accuracy']
            sum=sum+b
        Mean=sum/1
        print(conf.personality[i],"Accuracy:",Mean) 


if __name__ == "__main__":
    
    np.random.seed(10)
    low_repetitions = 0
    num_repetitions = conf.max_n_iter
    verbosity = 0
    shuffle_labels = False
    trait_list = range(0, conf.n_traits)       
       
    predict_all()
    F1_score()
    accuracy()