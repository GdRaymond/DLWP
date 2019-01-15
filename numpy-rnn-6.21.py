import numpy as np
time_steps=20
input_features=5
output_features=10
inputs=np.random.random((time_steps,input_features))
state_t=np.zeros((output_features,))
W=np.random.random((output_features,input_features))
print('W={0}'.format(W))
U=np.random.random((output_features,output_features))
b=np.random.random((output_features))

successive_outputs=[]
for numb,input_t in enumerate(inputs):
    dot_t=np.dot(W,input_t)#(output_features,input_features).(input_features,)=(output_features,)
    print('{0}--{1}--{2}'.format(numb,input_t,dot_t))
    output_t=np.tanh(dot_t+np.dot(U,state_t)+b)
    successive_outputs.append(output_t)
    state_t=output_t
print('successive_outputs={0}'.format(successive_outputs))
final_output_sequence=np.concatenate(successive_outputs,axis=0)
print('final_ouput:',final_output_sequence)