from keras.datasets import imdb
from keras import preprocessing
max_features=10000
maxlen=100
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
#print('len={2},x_test={0} \n y_test={1}'.format(x_test,y_test,len(x_test)))
x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen) #get last 20 words
x_test=preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)
#print('after pading\n len={2},x_test={0} \n y_test={1}'.format(x_test,y_test,len(x_test)))

from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding

model=Sequential()
model.add(Embedding(10000,16,input_length=maxlen))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
model.summary()

history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_split=0.2)


