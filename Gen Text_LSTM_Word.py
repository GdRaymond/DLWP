import keras
path=keras.utils.get_file('./nietzsche.txt',origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text=open(path).readline().lower()
texts=[]
with open(path) as f:
    for line in f:
        texts.append(line)

print('text top 10',texts[:10])

from keras.preprocessing.text import Tokenizer
max_words=10000
maxlen=50
tokenizer=Tokenizer(num_words=max_words,char_level=False,split=' ')
tokenizer.fit_on_texts(texts)
sequenses=tokenizer.sequences_to_texts(texts)
print('sequenses len={0}, content={1}'.format(len(sequenses),sequenses[10000:10050]))
word_index=tokenizer.word_index
print('found word_index len={0},word_index={1}'.format(len(word_index),word_index))