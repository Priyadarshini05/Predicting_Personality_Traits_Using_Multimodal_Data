'''
Affective Computing with AMIGOS Dataset
'''

from argparse import ArgumentParser
import os
import time
import numpy as np
import pandas as pd
import conf
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier as GBDT
from sklearn.model_selection import KFold,GroupKFold,LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def fisher_idx(num, features, labels):
    ''' Get idx sorted by fisher linear discriminant '''
    labels = np.array(labels)
    labels0 = np.where(labels < 1)
    labels1 = np.where(labels > 0)
    labels0 = np.array(labels0).flatten()
    labels1 = np.array(labels1).flatten()
    features0 = np.delete(features, labels1, axis=0)
    features1 = np.delete(features, labels0, axis=0)
    mean_features0 = np.mean(features0, axis=0)
    mean_features1 = np.mean(features1, axis=0)
    std_features0 = np.std(features0, axis=0)
    std_features1 = np.std(features1, axis=0)
    std_sum = std_features1**2 + std_features0**2
    fisher = (abs(mean_features0 - mean_features1)) / std_sum
    fischer1=meth_agn_v2(fisher, 0.1)
    sorted_feature_idx = fischer1[::-1]  # arrange from large to small
    return sorted_feature_idx[:num]

#Selectng features with J value > 0.1 
def meth_agn_v2(x, thresh):
    idx, = np.where(x > thresh)
    return idx[np.argsort(x[idx])]
          
def main():
    ''' Main function '''
    
    # read extracted features by loading the necessary file
    amigos_data = np.loadtxt('Data/Features_All_20s.csv', delimiter=',',skiprows=1) 
    print('Input data size:',amigos_data.shape)
    
    # read output personality by loading the necessary file
    labels = pd.read_csv("Data/Final_Personality_20s.csv")
    print('Output size:',labels.shape)
    for traits in range(0,conf.n_traits):  
        ext_labels = labels[conf.personality[traits]].values

        # setup kfold cross validator
        outer_kfold = GroupKFold(n_splits=conf.outer_folds)
        inner_kfold = KFold(n_splits=conf.inner_folds)
        num=conf.num #no of features
        sel_num=conf.sel_num #no of selected features
        
        # setup classifier
        ext_clf = SVC(C=conf.C, kernel=conf.kernel)  
        #ext_clf = xgb.XGBClassifier(max_depth=3,learning_rate=0.1,n_estimators=11,silent=True,objective="binary:logistic",nthread=-1,gamma=0,min_child_weight=.96,max_delta_step=0,subsample=1,colsample_bytree=1,colsample_bylevel=.99,reg_alpha=0,reg_lambda=.85,scale_pos_weight=1,base_score=.5,seed=13)
        
        
        # initialize index list
        Inner_Index_sorted=[] 
        Inner_Index_highest=[]
        
        # initialize outer history list 
        train_ext_accuracy_history2 = []
        train_ext_f1score_history2 = []
        val_ext_accuracy_history2 = []
        val_ext_f1score_history2 = []
        
        ext_idx_history = np.zeros(amigos_data.shape[1])
        
        start_time = time.time()
        Index = pd.read_csv("Data/ids_20s.csv")
        Index_labels = Index['Index'].values
        
        for outer_idx, (train_idx, val_idx) in enumerate(outer_kfold.split(amigos_data,groups=Index_labels)):
        
            #print('Outer_cv:',outer_idx + 1, 'Fold Start')
            train_data1, val_data1 = amigos_data[train_idx], amigos_data[val_idx]
            train_ext_labels, val_ext_labels = ext_labels[train_idx], ext_labels[val_idx]
           
            
            #Principle Componant Analysis
            ext_pca = PCA(n_components = conf.pca_channels)
            ext_pca.fit(train_data1,train_ext_labels)
            train_data = ext_pca.transform(train_data1)
            val_data = ext_pca.transform(val_data1)
            
             # initialize inner  history list
            train_ext_f1score_history1 = []
            val_ext_f1score_history1 = []
            index_list=[]

            for inner_idx, (inner_train_idx, inner_val_idx) in enumerate(inner_kfold.split(train_data)):
                #print('inner_cv:',inner_idx + 1, 'Fold Start')
                inner_train_data, inner_val_data = train_data[inner_train_idx], train_data[inner_val_idx]
                inner_train_ext_labels, inner_val_ext_labels = train_ext_labels[inner_train_idx], train_ext_labels[inner_val_idx]
                
                # fit feature selection
            
                #fisher J<0.1 channels are discorded
                ext_idx = fisher_idx(num, inner_train_data, inner_train_ext_labels)
                index_list.append(ext_idx)
                
                inner_train_ext_data = inner_train_data[:, ext_idx]
                inner_val_ext_data = inner_val_data[:, ext_idx]

                #record feature selection history
                for i in ext_idx:
                    ext_idx_history[i] += 1

                
                #pearson correlation
                df1=pd.DataFrame(inner_train_ext_data)
                df2=pd.DataFrame(inner_val_ext_data)
                #find correlation
                correlation_mat1 = df1.corr()
                correlation_mat2 = df2.corr()
                # find correlation between the column with highest J value and the rest of the columns
                cor1=correlation_mat1[0].sort_values(ascending=False)
                cor2=correlation_mat2[0].sort_values(ascending=False)

                # p>0.5 columns  are discorded                
                for j in range(1,len(df1.columns)): #Since column length of train and test are same we use only one for loop
                    if cor1[j] > 0.5 :
                        df1=df1.drop(columns=[j])
                    if cor2[j] > 0.5 :
                        df2=df2.drop(columns=[j])

                inner_train_ext_data = df1.to_numpy()  
                inner_val_ext_data = df2.to_numpy()  
                

                # fit classifier
                ext_clf.fit(inner_train_ext_data, inner_train_ext_labels)
                
                # predict traits
                inner_train_ext_predict_labels = ext_clf.predict(inner_train_ext_data)
                inner_val_ext_predict_labels = ext_clf.predict(inner_val_ext_data)

                # metrics calculation (accuracy and f1 score)
                train_ext_f1score1 = f1_score(inner_train_ext_labels, inner_train_ext_predict_labels, average='macro')
                val_ext_f1score1 = f1_score(inner_val_ext_labels, inner_val_ext_predict_labels, average='macro')

                train_ext_f1score_history1.append(train_ext_f1score1)
                val_ext_f1score_history1.append(val_ext_f1score1)

            #finding the index list with highest f1_score in inner loop
            Inner_Highest_f1_score_no=(np.argsort(val_ext_f1score_history1)[::-1])[0]
            Highest_index=index_list[Inner_Highest_f1_score_no]
            Inner_Index_highest.append(Highest_index)

            #sorting the index list and selecting the [1:f] most import features
            sort_ext_idx_history = np.argsort(ext_idx_history)[::-1][:sel_num]
            print('sorted_history:',type(sort_ext_idx_history),sort_ext_idx_history)
            Inner_Index_sorted.append(sort_ext_idx_history)
                       
            #only important features
            train_ext_data = train_data[:, sort_ext_idx_history]
            val_ext_data = val_data[:, sort_ext_idx_history]

            # fit classifier
            ext_clf.fit(train_ext_data, train_ext_labels)
            
            # predict traits
            train_ext_predict_labels = ext_clf.predict(train_ext_data)
            val_ext_predict_labels = ext_clf.predict(val_ext_data)

            # metrics calculation (accuracy and f1 score)
            train_ext_accuracy2 = accuracy_score(train_ext_labels, train_ext_predict_labels)
            train_ext_f1score2 = f1_score(train_ext_labels, train_ext_predict_labels, average='macro')
            val_ext_accuracy2 = accuracy_score(val_ext_labels, val_ext_predict_labels)
            val_ext_f1score2 = f1_score(val_ext_labels, val_ext_predict_labels, average='macro')
            train_ext_accuracy_history2.append(train_ext_accuracy2)
            train_ext_f1score_history2.append(train_ext_f1score2)
            val_ext_accuracy_history2.append(val_ext_accuracy2)
            val_ext_f1score_history2.append(val_ext_f1score2)
   
            
        print('\nDone. Duration: ', time.time() - start_time)

        print('\nAverage Training Result')
        print(conf.personality[traits],"=> Accuracy: {:.4f}, F1score: {:.4f}".format(
            np.mean(train_ext_accuracy_history2), np.mean(train_ext_f1score_history2)))
            
        print('Average Validating Result')
        print(conf.personality[traits]," => Accuracy: {:.4f}, F1score: {:.4f}".format(
            np.mean(val_ext_accuracy_history2), np.mean(val_ext_f1score_history2)))

        #store metrics in file
        with open('Classifier/SVM_train_history', 'a') as train_file:
             train_file.write("{}=> F1score:{}, Accuracy:{} \n".format(conf.personality[traits],np.mean(train_ext_f1score_history2),np.mean(train_ext_accuracy_history2)))
            
                 
        with open('Classifier/SVM_val_history', 'a') as val_file:
            val_file.write("{}=> F1score:{}, Accuracy:{}\n".format(conf.personality[traits],np.mean(val_ext_f1score_history2),np.mean(val_ext_accuracy_history2)))

                    
        
if __name__ == '__main__':

    main()

