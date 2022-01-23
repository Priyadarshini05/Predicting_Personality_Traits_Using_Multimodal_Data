from argparse import ArgumentParser
import os,glob
import warnings
import numpy as np
from biosppy.signals import ecg
from scipy.stats import skew, kurtosis
import scipy.io as spio
import pandas as pd
from PyEMD import EMD
from scipy.signal import butter, lfilter, filtfilt, welch
from sklearn.feature_selection import f_classif
from sklearn import preprocessing as pre
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

def detrend(data):
    ''' Detrend data with EMD '''
    emd = EMD()
    imfs = emd(data)
    detrended = np.sum(imfs[:int(imfs.shape[0] / 2)], axis=0)
    trend = np.sum(imfs[int(imfs.shape[0] / 2):], axis=0)

    return detrended, trend

def gsr_preprocessing(signals):
    ''' Preprocessing for GSR signals '''
    der_signals = np.gradient(signals)
    con_signals = 1.0 / signals
    nor_con_signals = (con_signals - np.mean(con_signals)) / np.std(con_signals)

    mean = np.mean(signals)
    der_mean = np.mean(der_signals)
    neg_der_mean = np.mean(der_signals[der_signals < 0])
    neg_der_pro = float(der_signals[der_signals < 0].size) / float(der_signals.size)

    local_min = 0
    for i in range(signals.shape[0] - 1):
        if i == 0:
            continue
        if signals[i - 1] > signals[i] and signals[i] < signals[i + 1]:
            local_min += 1

    # Using SC calculates rising time
    det_nor_signals, trend = detrend(nor_con_signals)
    lp_det_nor_signals = butter_lowpass_filter(det_nor_signals, 0.5, 128.)
    der_lp_det_nor_signals = np.gradient(lp_det_nor_signals)

    rising_time = 0
    rising_cnt = 0
    for i in range(der_lp_det_nor_signals.size - 1):
        if der_lp_det_nor_signals[i] > 0:
            rising_time += 1
            if der_lp_det_nor_signals[i + 1] < 0:
                rising_cnt += 1

    avg_rising_time = rising_time * (1. / 128.) / rising_cnt

    freqs, power = getfreqs_power(signals, fs=128., nperseg=signals.size, scaling='spectrum')
    power_0_24 = []
    for i in range(12):
        power_0_24.append(getBand_Power(freqs, power, lower=0 +(i * 0.2), upper=0.2 + (i * 0.2)))

    SCSR, _ = detrend(butter_lowpass_filter(nor_con_signals, 0.2, 128.))
    SCVSR, _ = detrend(butter_lowpass_filter(nor_con_signals, 0.08, 128.))
    
    
    zero_cross_SCSR = 0
    zero_cross_SCVSR = 0
    peaks_cnt_SCSR = 0
    peaks_cnt_SCVSR = 0
    peaks_value_SCSR = 0.
    peaks_value_SCVSR = 0.

    zc_idx_SCSR = np.array([], int)  # must be int, otherwise it will be float
    zc_idx_SCVSR = np.array([], int)
    for i in range(nor_con_signals.size - 1):
        if SCSR[i] * next((j for j in SCSR[i + 1:] if j != 0), 0) < 0:
            zero_cross_SCSR += 1
            zc_idx_SCSR = np.append(zc_idx_SCSR, i + 1)
        if SCVSR[i] * next((j for j in SCVSR[i + 1:] if j != 0), 0) < 0:
            zero_cross_SCVSR += 1
            zc_idx_SCVSR = np.append(zc_idx_SCVSR, i)

    for i in range(zc_idx_SCSR.size - 1):
        peaks_value_SCSR += np.absolute(SCSR[zc_idx_SCSR[i]:zc_idx_SCSR[i + 1]]).max()
        peaks_cnt_SCSR += 1
    for i in range(zc_idx_SCVSR.size - 1):
        peaks_value_SCVSR += np.absolute(SCVSR[zc_idx_SCVSR[i]:zc_idx_SCVSR[i + 1]]).max()
        peaks_cnt_SCVSR += 1

    zcr_SCSR = zero_cross_SCSR / (nor_con_signals.size / 128.)
    zcr_SCVSR = zero_cross_SCVSR / (nor_con_signals.size / 128.)

    mean_peak_SCSR = peaks_value_SCSR / peaks_cnt_SCSR if peaks_cnt_SCSR != 0 else 0
    mean_peak_SCVSR = peaks_value_SCVSR / peaks_cnt_SCVSR if peaks_value_SCVSR != 0 else 0

    features = [mean, der_mean, neg_der_mean, neg_der_pro, local_min, avg_rising_time] + power_0_24 + [zcr_SCSR, zcr_SCVSR, mean_peak_SCSR, mean_peak_SCVSR]
    return features


def read_dataset():
    cnt=0
    GSR_tmp=[]
   
    video_size=np.array([4,2,6,4,6,3,5,3,7,3,3,4,5,3,5,3]) #20s

    for p in glob.glob('/home/pushpalathane/dataset/Mat files/Data_Preprocessed_P??.mat'):
        data=spio.loadmat(p)
        cnt+=1
        #print(cnt)
        for j in range(0,len(video_size)):
            for k in range(0,video_size[j]):
                n1=k*(2560)
                n2=(k+1)*(2560)
                gsr_signals = data['joined_data'][0,j][n1:n2, -1]  
                gsr_features = gsr_preprocessing(gsr_signals)
                
                features = np.array(gsr_features)
                #Store the extracted features in list 
                GSR_tmp.append(features)
                    
        
        #Creating column names
        feature_gcr = ['mean_SR', 'mean_deri', 'mean_dif_neg', 'propo_neg', 'num_local_min', 'mean_rise', 'sp_0.0_0.2',
                       'sp_0.2_0.4', 'sp_0.4_0.6', 'sp_0.6_0.8', 'sp_0.8_1.0', 'sp_1.0_1.2', 'sp_1.2_1.4', 'sp_1.4_1.6',
                       'sp_1.6_1.8', 'sp_1.8_2.0', 'sp_2.0_2.2', 'sp_2.2_2.4', 'ZC_SCSR', 'ZC_SCVSR',
                       'mag_SCSR', 'mag_SCVSR']
       
        #Store the psd values into dataframe 
        GSR2=pd.DataFrame(GSR_tmp,columns=feature_gcr)
        print(GSR2.shape)
        GSR2.to_csv('Features_GSR_20s_un.csv',index= False)
        
    
read_dataset()

#Participant wise normalisation
import pandas as pd
import numpy as np
GSR={}
print(type(GSR))

df=pd.read_csv('Features_GSR_20s_un.csv')
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
    
    if not len(GSR):
        GSR=data
    else:
        GSR=pd.concat([GSR,data],ignore_index=True)


print(GSR.head)
GSR.to_csv(r"Data/Features_GSR_20s.csv",index= False)



