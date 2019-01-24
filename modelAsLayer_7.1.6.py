from keras import Input, layers,Model
from keras.applications import Xception

xception_base=Xception(weights=None,include_top=False)

left_input=Input(shape=(250,250,3))
right_input=Input(shape=(250,250,3))

left_feature=xception_base(left_input)
right_feature=xception_base(right_input)

merged_feature=layers.concatenate([left_feature,right_feature],axis=-1)
predictions=layers.Dense(1,activation='relu')(merged_feature)

model=Model([left_input,right_input],predictions)
model.summary()