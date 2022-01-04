
from argparse import ArgumentParser
import os,glob,re
import warnings
import numpy as np
from biosppy.signals import ecg
from scipy.stats import skew, kurtosis
import scipy.io as spio
import pandas as pd
from scipy.signal import butter, lfilter, filtfilt, welch
from sklearn.feature_selection import f_classif
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

def butter_highpass_filter(data, cutoff, fs, order=5):
    ''' Highpass filter '''
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=5):
    ''' Lowpass filter '''
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

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

def ecg_preprocessing(signals):
    ''' Preprocessing for ECG signals '''
    # some data have high peak value due to noise
    # signals , _ = detrend(signals)
    signals = butter_highpass_filter(signals, 1.0, 128.0)
    ecg_all = ecg.ecg(signal=signals, sampling_rate=128., show=False)
    rpeaks = ecg_all['rpeaks']  # R-peak location indices.

    # ECG
    freqs, power = getfreqs_power(signals, fs=128., nperseg=signals.size, scaling='spectrum')
    power_0_6 = []
    for i in range(60):
        power_0_6.append(getBand_Power(freqs, power, lower=0 + (i * 0.1), upper=0.1 + (i * 0.1)))

    IBI = np.array([])
    for i in range(len(rpeaks) - 1):
        IBI = np.append(IBI, (rpeaks[i + 1] - rpeaks[i]) / 128.0)

    heart_rate = np.array([])
    for i in range(len(IBI)):
        append_value = 60.0 / IBI[i] if IBI[i] != 0 else 0
        heart_rate = np.append(heart_rate, append_value)

    mean_IBI = np.mean(IBI)
    rms_IBI = np.sqrt(np.mean(np.square(IBI)))
    std_IBI = np.std(IBI)
    skew_IBI = skew(IBI)
    kurt_IBI = kurtosis(IBI)
    per_above_IBI = float(IBI[IBI > mean_IBI + std_IBI].size) / float(IBI.size)
    per_below_IBI = float(IBI[IBI < mean_IBI - std_IBI].size) / float(IBI.size)
    
    # IBI
    
    mean_heart_rate = np.mean(heart_rate)
    std_heart_rate = np.std(heart_rate)
    skew_heart_rate = skew(heart_rate)
    kurt_heart_rate = kurtosis(heart_rate)
    per_above_heart_rate = float(heart_rate[heart_rate >
                                            mean_heart_rate + std_heart_rate].size) / float(heart_rate.size)
    per_below_heart_rate = float(heart_rate[heart_rate <
                                            mean_heart_rate - std_heart_rate].size) / float(heart_rate.size)

    features = power_0_6+[rms_IBI, mean_IBI , std_IBI, skew_IBI, kurt_IBI, per_above_IBI, per_below_IBI] + [mean_heart_rate, std_heart_rate, skew_heart_rate, kurt_heart_rate, per_above_heart_rate, per_below_heart_rate]
   
    

    return features


def read_dataset():
    cnt=0
    ECG_tmp=[]
    
    video_size=np.array([4,2,6,4,6,3,5,3,7,3,3,4,5,3,5,3]) #20s
   
    for p in glob.glob('/home/pushpalathane/dataset/Mat files/Data_Preprocessed_P??.mat'):
        data=spio.loadmat(p)
        cnt+=1
        for j in range(0,len(video_size)):
            for k in range(0,video_size[j]):
                n1=k*(2560)
                n2=(k+1)*(2560)
                ecg_signals = data['joined_data'][0,j][n1:n2, 14]  # Column 14 or 15
                ecg_features = ecg_preprocessing(ecg_signals)
                features = np.array(ecg_features)
                         
                ECG_tmp.append(features)
               
        #Creating column names
        feature_ecg = ['ecg_sp_0.0_0.1', 'ecg_sp_0.1_0.2', 'ecg_sp_0.2_0.3', 'ecg_sp_0.3_0.4', 'ecg_sp_0.4_0.5', 
                       'ecg_sp_0.5_0.6','ecg_sp_0.6_0.7', 'ecg_sp_0.7_0.8', 'ecg_sp_0.8_0.9', 'ecg_sp_0.9_1.0', 
                       'ecg_sp_1.0_1.1', 'ecg_sp_1.1_1.2', 'ecg_sp_1.2_1.3', 'ecg_sp_1.3_1.4', 'ecg_sp_1.4_1.5',
                       'ecg_sp_1.5_1.6', 'ecg_sp_1.6_1.7', 'ecg_sp_1.7_1.8', 'ecg_sp_1.8_1.9', 'ecg_sp_1.9_2.0', 
                       'ecg_sp_2.0_2.1', 'ecg_sp_2.1_2.2', 'ecg_sp_2.2_2.3', 'ecg_sp_2.3_2.4','ecg_sp_2.4_2.5', 
                       'ecg_sp_2.5_2.6', 'ecg_sp_2.6_2.7', 'ecg_sp_2.7_2.8', 'ecg_sp_2.8_2.9', 'ecg_sp_2.9_3.0', 
                       'ecg_sp_3.0_3.1', 'ecg_sp_3.1_3.2', 'ecg_sp_3.2_3.3', 'ecg_sp_3.3_3.4', 'ecg_sp_3.4_3.5',
                       'ecg_sp_3.5_3.6', 'ecg_sp_3.6_3.7', 'ecg_sp_3.7_3.8', 'ecg_sp_3.8_3.9','ecg_sp_3.9_4.0',
                       'ecg_sp_4.0_4.1', 'ecg_sp_4.1_4.2', 'ecg_sp_4.2_4.3', 'ecg_sp_4.3_4.4', 'ecg_sp_4.4_4.5', 
                       'ecg_sp_4.5_4.6', 'ecg_sp_4.6_4.7', 'ecg_sp_4.7_4.8', 'ecg_sp_4.8_4.9',  'ecg_sp_4.9_5.0', 
                       'ecg_sp_5.0_5.1', 'ecg_sp_5.1_5.2', 'ecg_sp_5.2_5.3', 'ecg_sp_5.3_5.4','ecg_sp_5.4_5.5', 
                       'ecg_sp_5.5_5.6', 'ecg_sp_5.6_5.7', 'ecg_sp_5.7_5.8', 'ecg_sp_5.8_5.9', 'ecg_sp_5.9_6.0',
                       'rms_pin', 'mean_pin', 'std_pin', 'ske_pin', 'kur_pin', 'usum_pin', 'dsum_pin',
                       'mean_HR', 'std_HR', 'ske_HR', 'kur_HR', 'usum_HR', 'dsum_HR']
       
        #Store the psd values into dataframe 
       
        ECG2=pd.DataFrame(ECG_tmp,columns=feature_ecg)
        print(ECG2.shape)
        ECG2.to_csv('Features_ECG_20s_un.csv', index= False)
       
        

    
read_dataset()
    

#Participant wise normalisation
import pandas as pd
import numpy as np
ECG={}
#print(type(ECG))

df=pd.read_csv('Features_ECG_20s_un.csv')
df=df.fillna(df.mean())
df.info()

for k in range(0,31):
    n1=k*66
    n2=(k+1)*66
    data=df.iloc[n1:n2,:]
    
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    data = (data - data_min) / (data_max - data_min)
    data = data * 2 - 1
    
    
    if not len(ECG):
        ECG=data
    else:
        ECG=pd.concat([ECG,data],ignore_index=True)

   
print(ECG.head)
ECG.to_csv(r"Data/Features_ECG_20s.csv",index= False)





