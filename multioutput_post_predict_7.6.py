from keras import layers,Input,Model

dict_post=50000
num_income_group=10
input_post=Input(shape=(None,),dtype='int32',name='posts')
embeded_post=layers.Embedding(dict_post,256)(input_post)
x=layers.Conv1D(128,5,activation='relu')(embeded_post)
x=layers.MaxPool1D(5)(x)
x=layers.Conv1D(256,5,activation='relu')(x)
x=layers.Conv1D(256,5,activation='relu')(x)
x=layers.MaxPool1D(5)(x)
x=layers.Conv1D(256,5,activation='relu')(x)
x=layers.Conv1D(256,5,activation='relu')(x)
x=layers.GlobalMaxPooling1D()(x)
x=layers.Dense(128,activation='relu')(x)

age_prediction=layers.Dense(1,name='age')(x)
income_prediction=layers.Dense(num_income_group,activation='softmax',name='income')(x)
gender_prediction=layers.Dense(1,activation='sigmoid',name='gender')(x)
model=Model(input_post,[age_prediction,income_prediction,gender_prediction])
model.summary()

import numpy as np
num_post=20000
maxlen=500
posts=np.random.randint(1,dict_post,size=(num_post,maxlen))
ages=np.random.randint(15,85,size=(num_post,1))
print('top 10 ages',ages[:10])
incomes=np.zeros((num_post,num_income_group))
income_indicies=np.random.randint(0,num_income_group,size=(num_post))
for i,group in enumerate(income_indicies):
    incomes[i,group]=1
print('top 10 incomes:',incomes[:10])
genders=np.random.randint(0,2,size=(num_post,1))
print('top 10 genders:',genders[:10])

model.compile(optimizer='rmsprop',
              loss=['mse','categorical_crossentropy','binary_crossentropy'],
              loss_weights=[0.25,1.,10.])
model.fit(posts,[ages,incomes,genders],epochs=10,batch_size=64)


