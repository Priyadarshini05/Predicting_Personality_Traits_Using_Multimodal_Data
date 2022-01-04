#concate  feature files
import scipy.io as spio
import pandas as pd

data_EEG=pd.read_csv('Data/Features_EEG_20s.csv')
data_ECG=pd.read_csv('Data/Features_ECG_20s.csv')
data_GSR=pd.read_csv('Data/Features_GSR_20s.csv')
data_Gaze=pd.read_csv('Data/Features_Gaze_20s.csv')

data_all=pd.concat([data_EEG,data_ECG,data_GSR,data_Gaze],axis=1) # for all features
data_bio=pd.concat([data_EEG,data_ECG,data_GSR],axis=1) #for bio features

data_all.to_csv('Data/Features_All_20s.csv',index= False)
data_bio.to_csv('Data/Features_Bio_20s.csv',index= False)
