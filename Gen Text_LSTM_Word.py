import keras
path=keras.utils.get_file('./nietzsche.txt',origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
#text=open(path).readline().lower()
texts=[]
next_words=[]
with open(path) as f:
    #for line in f:
     #   texts.append
    text=f.read()
    l_text=text.replace('\n',' ').split()
    print('l_text len={0}, top 5310 is {1}'.format(len(l_text),l_text[:54]))

from keras.preprocessing.text import Tokenizer
max_words=10000
maxlen=50

for i in range(len(l_text)-maxlen):
    texts.append(' '.join(l_text[i:i+maxlen]))
    next_words.append(l_text[i+maxlen])
print('texts len={0}, top 2={1}'.format(len(texts),texts[:3]))
print('next words len={0}, top 2={1}'.format(len(next_words),next_words[:3]))

tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)
word_index=tokenizer.word_index
x_train=keras.preprocessing.sequence.pad_sequences(sequences,maxlen=maxlen)
y_train=[]
for word in next_words:
    y_train.append(word_index.get(word))
print('x_train first is ',x_train[:20])
print('y_train first is ',y_train[:20])
print('Found {0} unique tokens.'.format(len(word_index)))

for

from keras.models import Model,Sequential
from keras import layers
model=Sequential()
model.add(layers.Embedding(max_words,16,input_length=maxlen))
model.add(layers.LSTM(128))
model.add(layers.Dense(max_words,activation='softmax'))

optimizer=keras.optimizers.Optimizer(lr=0.1)
model.compile(optimizer-optimizer,loss='categorical_crossentropy')

import numpy as np
def sample(predics,tempreture=1.0):
    predics=np.asarray(predics).astype('float64')
    predics=np.log(predics)/tempreture
    exp_predics=np.exp(predics)
    predics=exp_predics/np.sum(exp_predics)
    probas=np.random.multinomial(1,predics,1)
    return np.argmax(probas)

for epoch in range(2):
    print('epoch =1',epoch)
    model.fit(x_train, y_train, batch_size=128, epoch=1)
    for tempreture in [1.0,0.5]:
        print('-tempreture=',tempreture)
        start_text=texts[:maxlen]
        print('--start_text=',start_text)

