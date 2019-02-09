import keras
path=keras.utils.get_file('./nietzsche.txt',origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
#text=open(path).readline().lower()
texts=[]
next_words=[]
with open(path) as f:
    #for line in f:
     #   texts.append
    text=f.read()
    l_text=text.split()
    print('l_text len={0}, top 5310 is {1}'.format(len(l_text),l_text[:54]))

from keras.preprocessing.text import Tokenizer
max_words=10000
maxlen=50

for i in range(len(l_text)-maxlen):
    texts.append(' '.join(l_text[i:i+maxlen]))
    next_words.append(l_text[i+maxlen])
print('texts len={0}, top 2={1}'.format(len(texts),texts[:3]))
print('next words len={0}, top 2={1}'.format(len(next_words),next_words[:3]))

tokenizer=Tokenizer()
tokenizer.fit_on_texts(texts)
sequenses=tokenizer.sequences_to_texts(texts)
print('sequenses len={0}, content={1}'.format(len(sequenses),sequenses[:10]))
word_index=tokenizer.word_index
print('found word_index len={0},word_index={1}'.format(len(word_index),word_index))