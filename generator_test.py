import numpy as np
train_data=np.random.randint(1,100,size=(2000,100,50))

def generator_data(data,start,batch_size=128):
    print('From start point ',start)
    step=0
    start_point=start
    while 1:
        step+=1
        if start_point+batch_size>len(data):
            start_point=start
        end_point=start_point+batch_size
        if end_point>len(data):
            end_point=len(data)
        print('step={0}, data range=[{1}-{2}]'.format(step,start_point,end_point))
        output=data[start_point:end_point]
        start_point+=len(output)
        yield output

def generator_new_way(data,start,batch_size=128):
    print('New way from start point ',start)
    step=0
    start_point=start
    while 1:
        step+=1
        if start_point+batch_size>len(data):
            start_point=start
        rows=np.arange(start_point,min(start_point+batch_size,len(data)))
        print('step={0}, data range=[{1}-{2}]'.format(step,start_point,min(start_point+batch_size,len(data))))
        start_point+=len(rows)
        output=np.zeros((len(rows),100,50),dtype='int32')
        for j,row in enumerate(rows):
            output[j]=data[row]
        yield output

train_gen = generator_data(train_data, 0, 128)
train_gen_neway=generator_new_way(train_data,0,128)
for i in range(20):
    result=next(train_gen)
    print('get result len={0}, 1st is {1}\n'.format(len(result),result[1]))

for i in range(21):
    result=next(train_gen_neway)
    print('get result len={0}, 1st is {1}'.format(len(result),result[1]))
