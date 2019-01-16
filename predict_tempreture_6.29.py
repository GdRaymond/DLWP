import os
file_weather=os.path.abspath(os.path.join(os.getcwd(),'jena_climate/jena_climate_2009_2016.csv'))
print(file_weather)
f=open(file_weather)
lines=f.read().split('\n')
f.close()
print('get lines:',len(lines))
head=lines[0].split(',')
lines=lines[1:]
print('column title:',head)

import numpy as np
float_data=np.zeros((len(lines),len(head)-1))
for i,line in enumerate(lines):
    values=[float(x) for x in line.split(',')[1:]]
    float_data[i,:]=values
temp=float_data[:,1]

#Similar scale: blow transform the data to standard score, substract mean then devided by standard deviation,标准分数=数值减去平均值除以标准方差
mean=float_data[:20000].mean(axis=0)
float_data-=mean
std=float_data[:20000].std(axis=0)
print('std:',std)
float_data/=std

def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
    if max_index is None:
        max_index=len(data)-delay-1
    i=min_index+lookback
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>max_index:
                i=min_index+lookback
            rows=range(i,min(i+batch_size,max_index))
            i+=len(rows)
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets=np.zeros(len(rows))
        for j,row in enumerate(rows):
            indices=range(row-lookback,row,step)
            samples[j]=data[indices]
            targets[j]=data[row+delay][1]
        yield samples,targets

lookback=1440
step=6
delay=144
batch_size=128

train_gen=generator(float_data,lookback,delay,0,20000,True,batch_size,step)
val_gen=generator(float_data,lookback,delay,20001,30000,False,batch_size,step)
test_gen=generator(float_data,lookback,delay,30001,None,False,batch_size,step)

val_steps=(30000-20001-lookback)
test_steps=(len(float_data)-30001-lookback)

def evaluate_naive_method():
    batch_maes=[]
    for step in range(val_steps):
        samples,targets=next(val_gen)
        preds=samples[:,-1,1]
        mae=np.mean(np.abs(preds-targets))
        batch_maes.append(mae)
    error=np.mean(batch_maes)
    print('common sense error:{0}, in celsiums {1}'.format(error,error*std[1])) #std[1] is tempreture std

evaluate_naive_method()

from keras.models import Sequential
from keras.layers import Flatten,Dense,GRU
from keras.optimizers import RMSprop
model=Sequential()
'''
model.add(Flatten(input_shape=(lookback//step,float_data.shape[-1])))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
model.summary()
model.compile(optimizer=RMSprop(),loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)
'''
model.add(GRU(32,input_shape=(None,float_data.shape[-1])))
model.add(Dense(1))
model.summary()
model.compile(optimizer='rmsprop',loss='mae')
history=model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)

import matplotlib.pyplot as plt
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.figure()
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()