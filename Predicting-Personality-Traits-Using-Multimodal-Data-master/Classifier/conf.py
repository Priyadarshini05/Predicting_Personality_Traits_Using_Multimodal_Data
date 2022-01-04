import numpy as np

# global parameters
n_participants = 31
n_traits = 5
max_n_feat = 230
max_n_iter = 1
personality=['Extroversion','Agreeableness','Conscientiousness','Emotional Stability','Openness']

# cross validation paramters RF
n_inner_folds = 3
n_outer_folds = 5

# cross validation paramters SVM, XGB
inner_folds = 5
outer_folds = 10

#SVM Hyperparameters
C=0.25
kernel='linear'

#No of features to be selected
sel_num=10
num=50
pca_channels=50

#XGBoost Hyperparameters
gamma=10.0
min_child_weight=0.8
max_delta_step=0.2
base_score=0.5
subsample=0.84
colsample_bytree=0.92
reg_lambda=0.5
colsample_bylevel=1.0

# Random Forest Hyperparameters
tree_max_features = 15
tree_max_depth = 5
n_estimators = 100
max_n_jobs = 5

#def get_result_filename(annotation_val, trait, shuffle_labels, i, add_suffix=False):
def get_result_filename(trait, shuffle_labels, i, add_suffix=False): 
    filename = 'Classifier/trait_'+str(trait+1)
    if shuffle_labels:
        filename += '_rnd'
    if add_suffix:
        filename += '.npz'
    return filename




