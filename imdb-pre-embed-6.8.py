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

