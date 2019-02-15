from keras import models,layers

HEIGHT=50
WIDTH=60
CHANNELS=3
NUM_CLASS=10
model=models.Sequential()

model.add(layers.SeparableConv2D(32,3,activation='relu',input_shape=(HEIGHT,WIDTH,CHANNELS)))
model.add(layers.SeparableConv2D(64,3,activation='relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64,3,activation='relu'))
model.add(layers.SeparableConv2D(128,3,activation='relu'))
model.add(layers.MaxPooling2D(2))

model.add(layers.SeparableConv2D(64,3,activation='relu'))
model.add(layers.SeparableConv2D(128,3,activation='relu'))
model.add(layers.GlobalAveragePooling2D())

model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(NUM_CLASS,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy')

model.summary()