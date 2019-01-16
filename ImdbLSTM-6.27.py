from keras.datasets import imdb
from keras.preprocessing import sequence
dict_len=10000
maxlen=500
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=dict_len)
x_train=sequence.pad_sequences(x_train,maxlen)
x_test=sequence.pad_sequences(x_test,maxlen)

from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM
model=Sequential()
model.add(Embedding(dict_len,32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,batch_size=128,epochs=10,validation_split=0.2)

from matplotlib import pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation add')
plt.plot(epochs,loss,'ro',label='Training loss')
plt.plot(epochs,val_loss,'r',label='Validation loss')
plt.title('Accuracy and Loss')
plt.legend()
plt.show()