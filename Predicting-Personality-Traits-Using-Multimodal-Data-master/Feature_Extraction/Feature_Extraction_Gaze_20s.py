# to find pupil diameter
from math import sqrt 
  
def findPupilDiameter(x1, y1, x2, y2, x3, y3) : 
    x12 = x1 - x2;  
    x13 = x1 - x3;  
  
    y12 = y1 - y2;  
    y13 = y1 - y3;  
  
    y31 = y3 - y1;  
    y21 = y2 - y1;  
  
    x31 = x3 - x1;  
    x21 = x2 - x1;  
  
    # x1^2 - x3^2  
    sx13 = pow(x1, 2) - pow(x3, 2);  
  
    # y1^2 - y3^2  
    sy13 = pow(y1, 2) - pow(y3, 2);  
  
    sx21 = pow(x2, 2) - pow(x1, 2);  
    sy21 = pow(y2, 2) - pow(y1, 2);  
  
    f = (((sx13) * (x12) + (sy13) * 
          (x12) + (sx21) * (x13) + 
          (sy21) * (x13)) // (2 * 
          ((y31) * (x12) - (y21) * (x13)))); 
              
    g = (((sx13) * (y12) + (sy13) * (y12) + 
          (sx21) * (y13) + (sy21) * (y13)) // 
          (2 * ((x31) * (y12) - (x21) * (y13))));  
  
    c = (-pow(x1, 2) - pow(y1, 2) - 
         2 * g * x1 - 2 * f * y1);  
  
    # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0  
    # where centre is (h = -g, k = -f) and  
    # radius r as r^2 = h^2 + k^2 - c  
    h = -g;  
    k = -f;  
    sqr_of_r = h * h + k * k - c;  
  
    # r is the radius  
    r = round(sqrt(sqr_of_r), 5);  
  
    #print("Centre = (", h, ", ", k, ")");  
    #print("Radius = ", r);  
    
    return 2*r
  
# This code is adapted from the code of Ryuga in tutorials point


def gaze_preprocessing(Gaze):
    
    # compute right eye gaze information statistics
    min_dr = (min(Gaze['diameter_right']))
    max_dr = (max(Gaze['diameter_right']))
    mean_dr = (st.mean(Gaze['diameter_right']))
    median_dr = (st.median(Gaze['diameter_right']))
    std_dr = (st.stdev(Gaze['diameter_right']))

    min_xr = (min(Gaze['gaze_1_x']))
    max_xr = (max(Gaze['gaze_1_x']))
    mean_xr = (st.mean(Gaze['gaze_1_x']))
    median_xr = (st.median(Gaze['gaze_1_x']))
    std_xr = (st.stdev(Gaze['gaze_1_x']))

    min_yr = (min(Gaze['gaze_1_y']))
    max_yr = (max(Gaze['gaze_1_y']))
    mean_yr = (st.mean(Gaze['gaze_1_y']))
    median_yr = (st.median(Gaze['gaze_1_y']))
    std_yr = (st.stdev(Gaze['gaze_1_y']))

    # compute left eye gaze information statistics
    min_dl = (min(Gaze['diameter_left']))
    max_dl = (max(Gaze['diameter_left']))
    mean_dl = (st.mean(Gaze['diameter_left']))
    median_dl = (st.median(Gaze['diameter_left']))
    std_dl = (st.stdev(Gaze['diameter_left']))

    min_xl = (min(Gaze['gaze_0_x']))
    max_xl = (max(Gaze['gaze_0_x']))
    mean_xl = (st.mean(Gaze['gaze_0_x']))
    median_xl = (st.median(Gaze['gaze_0_x']))
    std_xl = (st.stdev(Gaze['gaze_0_x']))

    min_yl = (min(Gaze['gaze_0_y']))
    max_yl = (max(Gaze['gaze_0_y']))
    mean_yl = (st.mean(Gaze['gaze_0_y']))
    median_yl = (st.median(Gaze['gaze_0_y']))
    std_yl = (st.stdev(Gaze['gaze_0_y']))

    features = [min_dr,max_dr,mean_dr,median_dr,std_dr,min_xr,max_xr,mean_xr,median_xr,std_xr,min_yr,max_yr,mean_yr,median_yr, std_yr,min_dl,max_dl,mean_dl,median_dl,std_dl,min_xl,max_xl,mean_xl,median_xl,std_xl,min_yl,max_yl,mean_yl,median_yl,std_yl]

    return features

            

from   random import shuffle
import glob
import pandas as pd
import sys
import numpy as np
import os
import statistics as st

UserID=['1', '2', '3', '4', '5', '6', '7', '10', '11', '13', '14', '15', '16', '17', '18', '19', '20',  '25', '26', '27','29', '30', '31','32','34','35','36','37','38','39','40']
video_id=['010','013','138','018','019','020','023','030','031','034','036','004','005','058','080','009']
#video_size=np.array([9,5,12,8,12,6,11,7,15,6,6,9,11,6,10,7]) #10s
video_size=np.array([4,2,6,4,6,3,5,3,7,3,3,4,5,3,5,3]) #20s
#video_size=np.array([3,1,4,2,4,2,3,2,5,2,2,3,3,2,3,2]) #30s

eye_gaze = []

for p in range(0,len(UserID)):
    x=int(UserID[p])
    #print(x)
    for j in range(0,len(video_id)):
        df_iter = pd.read_csv('home/pushpalathane/ws2020_nithyasreepriyadarshini/dataset/FACE VIDEO/P'+UserID[p]+'_'+video_id[j]+'_face.csv')
        rawGaze = df_iter[['frame', 'timestamp', 'confidence','gaze_0_x','gaze_0_y','gaze_1_x', 'gaze_1_y','eye_lmk_X_23','eye_lmk_X_25','eye_lmk_X_27','eye_lmk_X_50','eye_lmk_X_53','eye_lmk_X_55','eye_lmk_Y_23','eye_lmk_Y_25','eye_lmk_Y_27','eye_lmk_Y_50','eye_lmk_Y_53','eye_lmk_Y_55']]
                
        dr = []
        dl = []
        # compute pupil diameter for each frame
        for i in range(0,len(df_iter.index)): 
             
            xr1 = rawGaze.iloc[i]['eye_lmk_X_50']
            xr2 = rawGaze.iloc[i]['eye_lmk_X_53']
            xr3 = rawGaze.iloc[i]['eye_lmk_X_55']
        
            yr1 = rawGaze.iloc[i]['eye_lmk_Y_50']
            yr2 = rawGaze.iloc[i]['eye_lmk_Y_53']
            yr3 = rawGaze.iloc[i]['eye_lmk_Y_55']


            xl1 = rawGaze.iloc[i]['eye_lmk_X_23']
            xl2 = rawGaze.iloc[i]['eye_lmk_X_25']
            xl3 = rawGaze.iloc[i]['eye_lmk_X_27']
        
            yl1 = rawGaze.iloc[i]['eye_lmk_Y_23']
            yl2 = rawGaze.iloc[i]['eye_lmk_Y_25']
            yl3 = rawGaze.iloc[i]['eye_lmk_Y_27']
        
            dr.append(findPupilDiameter(xr1, yr1, xr2, yr2, xr3, yr3))
            dl.append(findPupilDiameter(xl1, yl1, xl2, yl2, xl3, yl3))
            
        rawGaze['diameter_right'] = dr
        rawGaze['diameter_left'] = dl
        
        for k in range(0,video_size[j]):
            if x<=32:
                m1=(500*k)+1 
                m2=(500*(k+1))+1
            else:
                m1=(1200*k)+1 
                m2=(1200*(k+1))+1
                       
            FinalGaze= rawGaze[m1:m2]
            #print('P and J and Length of dr:',UserID[p],video_id[j],FinalGaze.shape)
            gaze_features = gaze_preprocessing(FinalGaze)
            features=np.array(gaze_features)         
            eye_gaze.append(features)
            
            # Create the pandas DataFrame of these features
            df = pd.DataFrame(eye_gaze, columns = ['min_dr','max_dr','mean_dr','median_dr','std_dr','min_xr','max_xr','mean_xr','median_xr','std_xr','min_yr','max_yr','mean_yr','median_yr','std_yr','min_dl','max_dl','mean_dl','median_dl','std_dl','min_xl','max_xl','mean_xl','median_xl','std_xl','min_yl','max_yl','mean_yl','median_yl','std_yl']) 

    # write to csv file - gaze features for all training video files
    print(df.shape)
    df.to_csv('features_gaze_20s_un.csv',index= False)

print('success')

#normalisation
Gaze={}
print(type(Gaze))

df=pd.read_csv('features_gaze_20s_un.csv')
df=df.fillna(df.mean())
df.info()

for k in range(0,len(UserID)):
    n1=k*66
    n2=(k+1)*66
    data=df.iloc[n1:n2,:]
      
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    data = (data - data_min) / (data_max - data_min)
    data = data * 2 - 1
        
    if not len(Gaze):
        Gaze=data
    else:
        Gaze=pd.concat([Gaze,data],ignore_index=True)

print(Gaze.head)
Gaze.to_csv("Data/Features_Gaze_20s.csv",index= False)

