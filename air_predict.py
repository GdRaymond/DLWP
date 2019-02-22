import pandas
import datetime

def parse(x):
    return datetime.datetime.strptime(x,'%Y %m %d %H')

def clean_data(source_file,target_file):
    #load data
    dataset = pandas.read_csv(source_file, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0,
                              date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column name
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index_name = 'date'
    # make all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # sumarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv(target_file)

#clean_data('raw_air.csv','pollution.csv')
from matplotlib import pyplot as plt
def show_raw_data():
    dataset=pandas.read_csv('pollution.csv',header=0,index_col=0)
    values=dataset.values
    print('top 10 values:',values[:10])
    groups=[0,1,2,3,5,6,7]
    i=1
    plt.figure()
    for group in groups:
        plt.subplot(len(groups),1,i)
        plt.plot(values[:,group])
        plt.title(dataset.columns[group],y=0.5,loc='right')
        i+=1
    plt.show()
#show_raw_data()

import numpy as np
import sklearn

def format_data():
