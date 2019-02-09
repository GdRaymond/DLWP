from keras.datasets import imdb
from keras.preprocessing import sequence
dict_len=5000
maxlen=100
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=dict_len)
x_train=sequence.pad_sequences(x_train,maxlen)
x_test=sequence.pad_sequences(x_test,maxlen)

from keras import Input,layers,Model
input=Input(shape=(maxlen,))
x=layers.Embedding(dict_len,128,input_length=maxlen,name='embed')(input)
x=layers.Conv1D(64,7,activation='relu')(x)
x=layers.MaxPool1D(5)(x)
x=layers.Conv1D(128,7,activation='relu')(x)
x=layers.GlobalMaxPooling1D()(x)
output=layers.Dense(1)(x)
model=Model(input,output)
model.summary()

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
from keras.callbacks import TensorBoard
call_backs=[TensorBoard(log_dir='logs',histogram_freq=1,)]
model.fit(x_train,y_train,batch_size=128,epochs=20,validation_split=0.2,callbacks=call_backs)