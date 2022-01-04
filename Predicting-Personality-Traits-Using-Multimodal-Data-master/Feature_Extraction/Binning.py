import pandas as pd
import sklearn as sk
from sklearn import preprocessing
import numpy as np

    
def binning(trait):    
    extroversion=trait.values.reshape(-1,1)
    est=sk.preprocessing.KBinsDiscretizer(n_bins=2,encode='ordinal', strategy='quantile')
    est.fit(extroversion)
    y=est.transform(extroversion)
    return y

df = pd.read_excel ('/home/pushpalathane/dataset/Participants_Personality.xlsx',sheet_name='Personalities')
df=df.iloc[0:5,0:32].T
extraversion=df.iloc[1:32,0]
Agreeableness=df.iloc[1:32,1]
Conscientiousness=df.iloc[1:32,2]
EmotionalStability=df.iloc[1:32,3]
Creativity=df.iloc[1:32,4]

Ext=binning(extroversion)
print('binned extraversion')
Agr=binning(Agreeableness)
print('binned Agreeableness')
Con=binning(Conscientiousness)
print('binned Conscientiousness')
ES=binning(EmotionalStability)
print('binned EmotionalStability')
Cre=binning(Creativity)
print('binned Creativity')
 
array_tuple=(Ext,Agr,Con,ES,Cre) 
output=np.hstack(array_tuple)
binned_personality=pd.DataFrame(output,columns=['Extroversion','Agreeableness','Conscientiousness','Emotional_Stability','Creativity'])
binned_personality.to_csv('/Data/Binned_Personality.csv')
    
    
    
    





