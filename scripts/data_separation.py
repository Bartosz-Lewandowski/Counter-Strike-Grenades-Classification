import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import os

#read data
mirage = pd.read_csv('./data/train-grenades-de_mirage.csv')
inferno = pd.read_csv('./data/train-grenades-de_inferno.csv')

#concate data
complete_data = pd.concat([mirage,inferno])
#save concatanate data to csv
complete_data.to_csv('./data/complete_data.csv', index = False)

#load complete dataset
data = pd.read_csv('./data/complete_data.csv')

#function to plot countplot
#I use it to show if dataset is balanced or not
def plot_count(feature, data):
    plt.figure(figsize = (7,7))
    ax = sns.countplot(x = feature, data = data,  palette='colorblind')
    height = sum([p.get_height() for p in ax.patches])
    for p in ax.patches:
            ax.annotate(f'{100*p.get_height()/height:.2f} %', (p.get_x()+0.3, p.get_height()+5),animated=True)

#Plotting countplot
plot_count('LABEL', data)
plt.savefig('./plots/complete_dataset_label_proportion.png')

#Separate dataset to label and features
X = data.loc[:, data.columns != 'LABEL']
y = data['LABEL']

#separate to train and test datasets
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state = 333)




#concat datasets
train_data = pd.concat([x_train,y_train], axis = 1)

#check proportions of new datasets
plot_count('LABEL', train_data)
plt.savefig('./plots/train_label_proportion.png')

#The datasets are separated correctly (proportions are maintained)
#save those datasets to csv files for future use

train_data.to_csv('./data/train.csv', index = False)
x_test.to_csv('./data/test.csv', index = False)
y_test.to_csv('./data/test_labels.csv', index = False)