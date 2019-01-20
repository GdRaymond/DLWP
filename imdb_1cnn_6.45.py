from keras.datasets import imdb
from keras.preprocessing import sequence

dict_len=10000
maxlen=500
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=dict_len)
x_train=sequence.pad_sequences(x_train,maxlen)
x_test=sequence.pad_sequences(x_test)

from keras.models import Sequential
from keras.layers import Embedding,Conv1D,MaxPool1D,GlobalMaxPooling1D,Dense
from keras.optimizers import RMSprop

model=Sequential()
model.add(Embedding(dict_len,32,input_length=maxlen))
model.add(Conv1D(32,7,activation='relu'))
model.add(MaxPool1D(5))
model.add(Conv1D(32,7,activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1))
model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

from matplotlib import pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='training acc')
plt.plot(epochs,val_acc,'b',label='validation acc')
plt.title('Accuracy')
plt.legend()
plt.show()
