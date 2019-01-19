from keras import layers,Model,Input
dict_text=10000
dict_question=10000
dict_answer=500

input_text=Input(shape=(None,),dtype='int32',name='text')
embed_text=layers.Embedding(dict_text,64)(input_text)
encode_text=layers.LSTM(32)(embed_text)

input_question=Input(shape=(None,),dtype='int32',name='question')
embed_question=layers.Embedding(dict_question,64)(input_question)
encode_question=layers.LSTM(16)(embed_question)

concated=layers.concatenate([encode_text,encode_question],axis=-1)
answers=layers.Dense(dict_answer,activation='softmax')(concated)

model=Model([input_text,input_question],answers)
model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

import numpy as np
num_samples=10000
maxlen=100
texts=np.random.randint(1,dict_text+1,size=(num_samples,maxlen))
questions=np.random.randint(1,dict_question+1,size=(num_samples,maxlen))
answers=np.random.randint(0,2,size=(num_samples,dict_answer))
history=model.fit([texts,questions],answers,batch_size=132,epochs=20)

