import matplotlib.pyplot as plt
import pandas as pd
import os

# read data
os.getcwd()
train_path = r'/train_test_data/train'
label = ['0','1','2']
meta = pd.read_csv(r'train.csv')

print(len(meta))

# visualise 
plt.figure(figsize = (8,8))
plt.pie(meta.groupby('label').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

