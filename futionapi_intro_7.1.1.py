from keras import Input,layers
from keras.models import Model,Sequential
input_tensor=Input(shape=(64,))
x=layers.Dense(32,activation='relu')(input_tensor)
x=layers.Dense(32,activation='relu')(x)
output_tensor=layers.Dense(10,activation='softmax')(x)
model=Model(input_tensor,output_tensor)
print('model of manually assemble is:')
model.summary()

seq_model=Sequential()
seq_model.add(layers.Dense(32,activation='relu',input_shape=(64,)))
seq_model.add(layers.Dense(32,activation='relu'))
seq_model.add(layers.Dense(10,activation='softmax'))
print('Sequential model is ')
seq_model.summary()

model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

import numpy as np
x_train=np.random.random((1000,64))
print('x_train top 2',x_train[:2])
y_train=np.random.random((1000,10))
print('y_train top 2',y_train[:2])

model.fit(x_train,y_train,epochs=10,batch_size=128)
score=model.evaluate(x_train,y_train)
print('score=',score)
