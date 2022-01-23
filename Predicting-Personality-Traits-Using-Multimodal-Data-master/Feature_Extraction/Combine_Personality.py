# Transpose of csv 
import numpy as np
import pandas as pd
df3=pd.read_csv('Binned_Personality.csv').drop(['Unnamed: 0'],axis=1)
print(df3.shape)
df4=df3.T
print(df4.head)
df4.to_csv('P2.csv') 

#generating for individual participants
from sklearn import preprocessing as pre
import numpy as np
import pandas as pd

UserID=['1', '2', '3', '4', '5', '6', '7', '8','9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
 '24', '25', '26', '27', '28','29', '30', '31']

df2=np.zeros((66,5))
for k in range(0,len(UserID)):
    df=pd.read_csv('Data/P2.csv')
    for i in range(0,66):
        df2[i][0]=df.iloc[0][k+1]
        df2[i][1]=df.iloc[1][k+1]
        df2[i][2]=df.iloc[2][k+1]
        df2[i][3]=df.iloc[3][k+1]
        df2[i][4]=df.iloc[4][k+1]

    col=['Extroversion','Agreeableness','Conscientiousness','Emotional Stability','Creativity(openness)'] 
    person=pd.DataFrame(df2.reshape((66,5)),columns=col)
    print(UserID[k])
    person.to_csv('Personality_'+UserID[k]+'_20s.csv',index= False)
    
print('success')
    
#Combining all personality data

import numpy as np
import pandas as pd
UserID=['31', '30', '29', '28', '27', '26','25', '24', '23', '22', '21', '20','19','18', '17',
        '16', '15','14', '13','12','11','10','9','8', '7', '6', '5','4', '3', '2', '1']

        
def accumulate(N):
    M=read(N)
    if N== 0: 
        return M 
    else: 
        return M.append(accumulate(N-1),ignore_index=True) 
    

def read(N):
    df1=pd.read_csv('Personality_'+UserID[N]+'_20s.csv')
    print(UserID[N])
    return df1

d=accumulate(30)

print(d.shape)
d.to_csv('Data/Final_Personality_20s.csv',index= False)
print('success')




