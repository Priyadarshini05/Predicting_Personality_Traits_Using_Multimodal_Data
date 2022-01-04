import pandas as pd
import numpy as np
ids=[]
for i in range(0,31):
    for j in range(0,66):
        ids.append(i)
    df=pd.DataFrame(ids,columns=['Index'])
    df.to_csv('ids_20s.csv', index=False)