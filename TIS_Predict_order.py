#original list style|total_quantity|invoice_date|id|colour|total_quantity|size1|size2|size3|size4|size5|size6|size7|size8|size9|size10|size11|size12|size13|size14|size15|size16|size17|size18|size19|size20|size21|size22|size23|size24|size25|size26|size27|size28|size29|size30|packing_id

import os
file_name=os.path.abspath(os.path.join(os.getcwd(),'tis_data/all_packing.csv'))
all_data=[]
with open(file_name) as f:
    for line in f:
        all_data.append(line.split('|'))
    print('all_data len={0}, top 2 ={1}'.format(len(all_data),all_data[:2]))

#filter style and colour
STYLE='RM200CF'
COLOUR='NAVY'
DATE_START='2012-01-01'
all_data=all_data[1:] #delete title
#all_data=list(filter(lambda x:(x[0]==STYLE and x[4]==COLOUR),all_data)) #FILTER STYLE AND COLOUR
all_data=list(filter(lambda x:(x[0]==STYLE),all_data)) #FILTER STYLE ONLY
print('all_data filtered lend={0}, top3={1}'.format(len(all_data),all_data[:]))
all_data=list(map(lambda x:x[2:],all_data)) #delete style total_quantity
print('style',all_data)
import dateutil,datetime
new_all_data=[]
for line in all_data:
    date=datetime.datetime.strptime(line[0],'%Y-%M-%d')
    date_start=datetime.datetime.strptime(DATE_START,'%Y-%M-%d')
    delta=date-date_start
    new_line=[delta.days] #[1108]
    new_line.extend(line[4:-1]) #[days,size1...size31]
    new_all_data.append(new_line)
print('new_all_data=',new_all_data)

import numpy as np
from operator import itemgetter
new_all_data=sorted(new_all_data,key=itemgetter(0)) #order by date
x_float=np.asarray(new_all_data,dtype='float32')
print('x_data shape={0},data={1}'.format(x_float.shape,x_float))

from sklearn.preprocessing import scale
new_float_data=scale(x_float) #same fucntion as x-mean/std, but avoid error of dividied by 0
print('new_float_data=',new_float_data)

#normalization
mean=x_float.mean(axis=0)
x_float-=mean
std=x_float.std(axis=0) #calculate

steps=6 #every 6 orders as sequence
x_data=[]
y_data=[]
for i in range(x_float.shape[0]-steps):
    x_data.append(x_float[i:i+steps])
    y_data.append(x_float[i+steps][4]) #take size 3 as target
x_data=np.asarray(x_data)
y_data=np.asarray(y_data)
print('x_date shape={0}, data={1}'.format(x_data.shape,x_data))
print('y_date shape={0}, data={1}'.format(y_data.shape,y_data))

from keras.models import Sequential
from keras.layers import Dense,GRU,LSTM
model=Sequential()
model.add(LSTM(8,dropout=0.2,recurrent_dropout=0.2,return_sequences=True,input_shape=(None,x_float.shape[-1])))
model.add(LSTM(16,activation='relu',dropout=0.1,recurrent_dropout=0.5))
model.add(Dense(1))
model.compile(optimizer='rmsprop',loss='mae')
history=model.fit(x_data,y_data,epochs=50,validation_split=0.2)

val_loss=history.history['val_loss']
print('std=',std[4])
