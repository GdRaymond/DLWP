import numpy as np
import keras
path=keras.utils.get_file('./nietzsche.txt',origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
#text=open(path).readline().lower()
texts=[]
next_words=[]
with open(path) as f:
    text=f.read()
    l_text=text.split()
    print('l_text len={0}, top 5310 is {1}'.format(len(l_text),l_text[:54]))

from keras.preprocessing.text import Tokenizer
max_words=11000
maxlen=70

for i in range(len(l_text)-50):
    texts.append(' '.join(l_text[i:i+50]))
    next_words.append(l_text[i+50])
print('texts len={0}, top 2={1}'.format(len(texts),texts[:3]))
print('next words len={0}, top 2={1}'.format(len(next_words),next_words[:3]))

tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts)
sequenses=tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
index_word=tokenizer.index_word
print('found word_index len={0},word_index={1}'.format(len(word_index), word_index))
print('found index_word len={0}, index_word={1}'.format(len(index_word),index_word))
x_train=keras.preprocessing.sequence.pad_sequences(sequenses,maxlen=maxlen)
y_train=np.zeros((len(next_words),len(word_index)))
for i,word in enumerate(next_words):
    y_train[i,word_index.get(word,2)-1]=1

print('x_train len={0}, content={1}'.format(len(x_train),x_train[:10]))
print('next words len={0}, content={1}'.format(len(next_words),next_words[:10]))
print('y_train len={0}, content={1}'.format(len(y_train),y_train[:10]))

from keras.models import Sequential
from keras.layers import Embedding,LSTM,Dense
model=Sequential()
model.add(Embedding(max_words,32,input_length=maxlen))
model.add(LSTM(128))
model.add(Dense(len(word_index),activation='softmax'))
optimizer=keras.optimizers.RMSprop(lr=0.1)
model.compile(optimizer=optimizer,loss='categorical_crossentropy')
model.summary()

def sample(predics,tempreture):
    print('original predics = {0}, tempreture={1}'.format(predics,tempreture))
    predics=np.log(predics) / tempreture
    print('log and divide tempreture predics =',predics)
    exp_predics=np.exp(predics)
    print('exp_predics=',exp_predics)
    predics=exp_predics/np.sum(exp_predics)
    print('exp_predics/np.sum(exp_predics=',predics)
    probas=np.random.multinomial(1,predics,1)
    print('probas=',probas)
    max_index=np.argmax(probas)
    return max_index

import sys
for epoch in range(2):
    print('epoch ',epoch)
    model.fit(x_train,y_train,batch_size=128,epochs=1)
    start_index=0
    l_generated=l_text[start_index:start_index+maxlen]
    generated_word=' '.join(l_generated)
    print('start generated_text is ',' '.join(l_generated))
    encoded_words = []
    for word in l_generated:
        encoded_words.append(word_index.get(word))

    for tempreture in [1.0,0.5]:
        print('--tempreture = ',tempreture)
        for i in range(100): #generated 100 words
            predics=model.predict(np.expand_dims(encoded_words,axis=0))
            print('origin next word index=',np.argmax(predics))
            gen_next_word_index=sample(predics,tempreture)
            print('generated next word index=',gen_next_word_index)
            generated_word='{0} {1}'.format(generated_word,index_word.get(gen_next_word_index))
            encoded_words=encoded_words.append(gen_next_word_index)[1:]
        print('---new generated text is',generated_word)


