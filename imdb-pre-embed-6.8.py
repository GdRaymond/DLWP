import os
imdb_dir='./aclImdb'
train_dir=os.path.join(imdb_dir,'train')

labels=[]
texts=[]

for label_type in ['neg','pos']:
    dir_name=os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:]=='.txt':
            fullname=os.path.join(dir_name,fname)
            #print('reading file {0}'.format(fullname))
            with open(fullname,encoding='UTF-8') as f:
                for line in f:
                    #print('reading line {0}'.format(line))
                    texts.append(line)
            if label_type=='neg':
                labels.append(0)
            else:
                labels.append(1)
print('len(texts)={0}, texts={1}'.format(len(texts),texts[12490:12510]))
print('len(labels)={0},labels={1}'.format(len(labels),labels[12490:12510]))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen=100
training_samples=200
validation_samples=10000
max_words=10000

tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)
word_index=tokenizer.word_index
print('Found {0} unique tokens.'.format(len(word_index)))
data=pad_sequences(sequences,maxlen=maxlen)

labels=np.asarray(labels)
print('Shape of data tensor:',data.shape)
print('Shape of label tensor:',labels.shape)

indices=np.arrange(data.shape[0])
np.random.shuffle(indices)
data=data[indices]
labels=labels[indices]

x_train=data[:training_samples]
y_train=labels[:training_samples]
x_val=data[training_samples:training_samples+validation_samples]
y_val=labels[training_samples:training_samples+validation_samples]
print('x_train first 10:',x_train[:10])
print('y_train first 10:',y_train[:10])

glove_dir='./glove.6B'
embeddings_index={}
f=open(os.path.join(glove_dir,'glove.6B.100d.txt'))
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
f.close()

embedding_dim=100
embedding_matrix=np.zeros((max_words,embedding_dim))
for word, i in word_index.items():
    if i<max_words:
        embedding_vector=embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i]=embedding_vector

print('embedding_matrix first 10:',embedding_matrix[:10])

from keras.models import Sequential
from keras.layers import Embedding, Flatten,Dense
model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))
model.save_weights('pre_trained_glove_model.h5')

import matplotlib.