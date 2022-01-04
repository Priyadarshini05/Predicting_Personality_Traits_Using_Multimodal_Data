from argparse import ArgumentParser
import os,glob
import warnings
import numpy as np
from biosppy.signals import ecg
from scipy.stats import skew, kurtosis
import scipy.io as spio
import pandas as pd
from scipy import stats
from sklearn import preprocessing as pre 
from scipy.signal import butter, lfilter, filtfilt, welch
from sklearn.feature_selection import f_classif
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def getfreqs_power(signals, fs, nperseg, scaling):
    ''' Calculate power density or power spectrum density '''
    if scaling == "density":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='density')
        return freqs, power
    elif scaling == "spectrum":
        freqs, power = welch(signals, fs=fs, nperseg=nperseg, scaling='spectrum')
        return freqs, power
    else:
        return 0, 0


def getBand_Power(freqs, power, lower, upper):
    ''' Sum band power within desired frequency range '''
    low_idx = np.array(np.where(freqs <= lower)).flatten()
    up_idx = np.array(np.where(freqs > upper)).flatten()
    band_power = np.sum(power[low_idx[-1]:up_idx[0]])

    return band_power


def getFiveBands_Power(freqs, power):
    ''' Calculate 5 bands power '''
    theta_power = getBand_Power(freqs, power, 3, 7)
    slow_alpha_power = getBand_Power(freqs, power, 8, 10)
    alpha_power = getBand_Power(freqs, power, 8, 13)
    beta_power = getBand_Power(freqs, power, 14, 29)
    gamma_power = getBand_Power(freqs, power, 30, 47)

    return theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power


def eeg_preprocessing(signals):
    ''' Preprocessing for EEG signals '''
    trans_signals = np.transpose(signals)
    #trans_signals = np.transpose(signals)

    theta_power = []
    slow_alpha_power = []
    alpha_power = []
    beta_power = []
    gamma_power = []
    psd_list = [theta_power, slow_alpha_power, alpha_power, beta_power, gamma_power]

    theta_spec_power = []
    slow_alpha_spec_power = []
    alpha_spec_power = []
    beta_spec_power = []
    gamma_spec_power = []
    spec_power_list = [theta_spec_power, slow_alpha_spec_power,
                       alpha_spec_power, beta_spec_power, gamma_spec_power]

    theta_spa = []
    slow_alpha_spa = []
    alpha_spa = []
    beta_spa = []
    gamma_spa = []

    theta_relative_power = []
    slow_alpha_relative_power = []
    alpha_relative_power = []
    beta_relative_power = []
    gamma_relative_power = []

    for channel_signals in trans_signals:
        freqs, power = getfreqs_power(channel_signals, fs=128.,
                                      nperseg=channel_signals.size, scaling='density')
        psd = getFiveBands_Power(freqs, power)
        for band, band_list in zip(psd, psd_list):
            band_list.append(band)

        freqs_, power_ = getfreqs_power(channel_signals, fs=128.,
                                        nperseg=channel_signals.size, scaling='spectrum')
        spec_power = getFiveBands_Power(freqs_, power_)
        for band, band_list in zip(spec_power, spec_power_list):
            band_list.append(band)

    for i in range(7):
        theta_spa.append((theta_spec_power[i] - theta_spec_power[13 - i]) /
                         (theta_spec_power[i] + theta_spec_power[13 - i]))
        slow_alpha_spa.append((slow_alpha_spec_power[i] - slow_alpha_spec_power[13 - i]) /
                              (slow_alpha_spec_power[i] + slow_alpha_spec_power[13 - i]))
        alpha_spa.append((alpha_spec_power[i] - alpha_spec_power[13 - i]) /
                         (alpha_spec_power[i] + alpha_spec_power[13 - i]))
        beta_spa.append((beta_spec_power[i] - beta_spec_power[13 - i]) /
                        (beta_spec_power[i] + beta_spec_power[13 - i]))
        gamma_spa.append((gamma_spec_power[i] - gamma_spec_power[13 - i]) /
                         (gamma_spec_power[i] + gamma_spec_power[13 - i]))

    total_power = np.array(theta_power) + np.array(alpha_power) +         np.array(beta_power) + np.array(gamma_power)

    for i in range(trans_signals.shape[0]):
        theta_relative_power.append(theta_power[i] / total_power[i])
        slow_alpha_relative_power.append(slow_alpha_power[i] / total_power[i])
        alpha_relative_power.append(alpha_power[i] / total_power[i])
        beta_relative_power.append(beta_power[i] / total_power[i])
        gamma_relative_power.append(gamma_power[i] / total_power[i])

    features = theta_power + slow_alpha_power + alpha_power + beta_power + gamma_power + theta_spa + slow_alpha_spa + alpha_spa + beta_spa +         gamma_spa 

    return features


def read_dataset():
    cnt=0
    
    EEG_tmp=[]
    
    video_size=np.array([4,2,6,4,6,3,5,3,7,3,3,4,5,3,5,3]) #20s
    
    for p in glob.glob('/home/pushpalathane/dataset/Mat files/Data_Preprocessed_P??.mat'):
        data=spio.loadmat(p)
        cnt+=1
        for j in range(0,len(video_size)):
             for k in range(0,video_size[j]):
                n1=k*(2560)
                n2=(k+1)*(2560)
                eeg_signals = data['joined_data'][0,j][n1:n2, :14]
                eeg_features = eeg_preprocessing(eeg_signals)
                features = np.array(eeg_features)
                EEG_tmp.append(features)
            
        
        #Creating column names
        feature_eeg = ['theta_AF3', 'slow_alpha_AF3', 'alpha_AF3', 'beta_AF3', 'gamma_AF3', 'theta_AF4', 
                'slow_alpha_AF4', 'alpha_AF4', 'beta_AF4', 'gamma_AF4', 'theta_AF3_AF4',    'slow_alpha_AF3_AF4',
                'alpha_AF3_AF4', 'beta_AF3_AF4', 'gamma_AF3_AF4', 'theta_F7', 'slow_alpha_F7', 'alpha_F7', 
                'beta_F7', 'gamma_F7', 'theta_F8', 'slow_alpha_F8', 'alpha_F8', 'beta_F8', 'gamma_F8', 
                'theta_F7_F8', 'slow_alpha_F7_F8', 'alpha_F7_F8', 'beta_F7_F8', 'gamma_F7_F8', 'theta_F3', 
                'slow_alpha_F3', 'alpha_F3', 'beta_F3', 'gamma_F3', 'theta_F4', 'slow_alpha_F4', 'alpha_F4', 
         'beta_F4', 'gamma_F4', 'theta_F3_F4', 'slow_alpha_F3_F4', 'alpha_F3_F4', 'beta_F3_F4',  'gamma_F3_F4', 
            'theta_FC5', 'slow_alpha_FC5', 'alpha_FC5', 'beta_FC5', 'gamma_FC5', 'theta_FC6', 'slow_alpha_FC6',
                'alpha_FC6', 'beta_FC6', 'gamma_FC6', 'theta_FC5_FC6', 'slow_alpha_FC5_FC6', 'alpha_FC5_FC6', 
                'beta_FC5_FC6', 'gamma_FC5_FC6', 'theta_T7', 'slow_alpha_T7', 'alpha_T7', 'beta_T7', 'gamma_T7',
            'theta_T8', 'slow_alpha_T8', 'alpha_T8', 'beta_T8', 'gamma_T8', 'theta_T7_T8', 'slow_alpha_T7_T8',
                'alpha_T7_T8', 'beta_T7_T8', 'gamma_T7_T8', 'theta_P7', 'slow_alpha_P7', 'alpha_P7', 'beta_P7',
                'gamma_P7', 'theta_P8', 'slow_alpha_P8', 'alpha_P8', 'beta_P8', 'gamma_P8', 'theta_P7_P8', 
                'slow_alpha_P7_P8', 'alpha_P7_P8', 'beta_P7_P8', 'gamma_P7_P8', 'theta_O1', 'slow_alpha_O1',
            'alpha_O1', 'beta_O1', 'gamma_O1', 'theta_O2', 'slow_alpha_O2', 'alpha_O2', 'beta_O2', 'gamma_O2',
                     'theta_O1_O2', 'slow_alpha_O1_O2', 'alpha_O1_O2', 'beta_O1_O2', 'gamma_O1_O2']
       
        #Store the psd values into dataframe 
        EEG2=pd.DataFrame(EEG_tmp,columns=feature_eeg)
        #EEG2=pd.DataFrame(EEG_tmp.reshape((496,105)),columns=feature_eeg)
        print(EEG2.shape)
        EEG2.to_csv('Features_EEG_20s_un.csv',index= False)
        
    
read_dataset()
    

import pandas as pd
import numpy as np
EEG={}
#print(type(EEG))

df=pd.read_csv('Features_EEG_20s_un.csv')
df=df.fillna(df.mean())
df.info()

for k in range(0,31):
    n1=k*66
    n2=(k+1)*66
    data=df.iloc[n1:n2,:]
    #print(data)
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    data = (data - data_min) / (data_max - data_min)
    data = data * 2 - 1
    if not len(EEG):
        EEG=data
    else:
        EEG=pd.concat([EEG,data],ignore_index=True)

print(EEG.head)
EEG.to_csv(r"Data/Features_EEG_20s.csv",index= False)



