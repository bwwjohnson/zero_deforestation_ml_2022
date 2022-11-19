# -*- coding: utf-8 -*-

# importing all the required libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
import os
import pandas as pd
import matplotlib.pyplot as plt

# get current working directory for the path
path = os.getcwd()

#import training data as .csv file
df = pd.read_csv(path+'/train.csv')

#visualise piechart for percentage of images in each scenario prior to augmentation
plt.figure(figsize = (8,8))
plt.pie(df.groupby('label').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

#create a file id using the dataframe['example_path']
df['File_ID'] = df.example_path.str.extract('(\d+)')

#create a version number to keep track of the augmentations that have occured to each image
df['version_no'] = np.zeros([len(df)])

#create a dataframe for each of the three scenarios
df_0 = df.copy(deep=True)
df_0.drop(df_0[df_0['label']!=0].index,inplace=True) #drop all images that are not scenario 0
df_0.reset_index(inplace=True) #reset index

#repeat above for scenario 1 & 2
df_1 = df.copy(deep=True)
df_1.drop(df_1[df_1['label']!=1].index,inplace=True)
df_1.reset_index(inplace=True)
df_2 = df.copy(deep=True)
df_2.drop(df_2[df_2['label']!=2].index,inplace=True)
df_2.reset_index(inplace=True)


#create a function that will perform rotaions and flipping to the images
def percentage_matcher(n_iter,df,df_0,df_1,df_2,path,outpath):
  '''change the percentages of files in each scenario to be similar to 2 dp.
  input       n_iter: numbner of iterations for the for loop.
  input           df: dataframe of all training data.
  input         df_0: dataframe of scenario 0.
  input         df_1: dataframe of scenario 1.
  input         df_2: dataframe of scenario 2.
  input         path: path to folder location to image file.
  input      outpath: path to where the new augmented images are to be saved. 
  return     lengths: the length of each directory.
  return percentages: the percentage of evenets in each directory.
  return          df: dataframe of augmented training data.'''

  len_0 = len(df_0) #calculate the number of images associated with scenarion 0
  len_1 = len(df_1) #calculate the number of images associated with scenarion 0
  len_2 = len(df_2) #calculate the number of images associated with scenarion 0
  lengths = np.array([len_0,len_1,len_2]) #put the starting number of images for each scenario into an array
  percentages = (lengths)/np.sum(lengths) #calculate the starting percentages of each image in the three scenarios
  folders = np.array(['0_img','1_img','2_img']) #create a list to be used when selecting images from different scenarios
  
  for i in range(n_iter):
    min_length = min(lengths) #find scenario with miniumum number of images
    boolarr = lengths == min_length #identify scenario with least events
    lengths[boolarr] = lengths[boolarr] + 5 #add to the scenario with the least number of images, the 5 represents the 5 augmentations conducted below
    percentages = (lengths)/np.sum(lengths) #calculate percentages
    
    if folders[boolarr] == '0_img': #determine which scenario to choose from
      sample_img_no = np.array(df_0.sample()) #select random image
    elif folders[boolarr] == '1_img': #determine which scenario to choose from
      sample_img_no = np.array(df_1.sample()) #select random image
    elif folders[boolarr] == '2_img': #determine which scenario to choose from
      sample_img_no = np.array(df_2.sample())#select random image
    
    #set image path to locate the image to be augmented
    image_path = '{h}/{p}.png'.format(h=path,k=folders[boolarr][0],p=sample_img_no[0][-2])
    
    #load in the image
    img = io.imread(image_path)
    #display the image
    #io.imshow(img)
    
    #assign the rotations
    rotations = [90,180,270]
    
    for j in range(len(rotations)):
      rotated = rotate(img, angle=rotations[j], mode = 'wrap') #rotate the image
      sample_img_no[0][-1] +=1 #create a version number for this augmentation
      sample_img_no_version = sample_img_no[0][1:] #select the part to be added to the orignial image dataframe
      df_ev = pd.DataFrame([sample_img_no_version],columns=['label','latitude','longitude','year','example_path','File_ID','version_no']) #create dataframe for this augmented image
      df_ev['example_path'] = 'train_test_data/train/{p}_{q}.png'.format(p=df_ev['File_ID'].iloc[-1],q=int(df_ev['version_no'].iloc[-1])) #set the exmaple path in the dataframe
      df = pd.concat([df,df_ev],ignore_index=True) #combine the dataframe to the original one with all images in
      io.imsave('{h}/{p}_{q}.png'.format(h=outpath,k=folders[boolarr][0],p=df['File_ID'].iloc[-1],q=int(df['version_no'].iloc[-1])),arr=rotated) #save the new augmented image
    
    flipLR = np.fliplr(img) #flip the image hoizontally
    sample_img_no[0][-1] +=1 #create a version number for this augmentation
    sample_img_no_version = sample_img_no[0][1:] #select the part to be added to the orignial image dataframe
    df_ev = pd.DataFrame([sample_img_no_version],columns=['label','latitude','longitude','year','example_path','File_ID','version_no']) #create dataframe for this augmented image
    df_ev['example_path'] = 'rain_test_data/train/{p}_{q}.png'.format(p=df_ev['File_ID'].iloc[-1],q=int(df_ev['version_no'].iloc[-1])) #set the exmaple path in the dataframe
    df = pd.concat([df,df_ev],ignore_index=True)  #combine the dataframe to the original one with all images in
    io.imsave('{h}/{p}_{q}.png'.format(h=outpath,k=folders[boolarr][0],p=df['File_ID'].iloc[-1],q=int(df['version_no'].iloc[-1])),arr=flipLR) #save the new augmented image
    
    flipUD = np.flipud(img) #flip the image vertically
    sample_img_no[0][-1] +=1 #create a version number for this augmentation
    sample_img_no_version = sample_img_no[0][1:] #select the part to be added to the orignial image dataframe
    df_ev = pd.DataFrame([sample_img_no_version],columns=['label','latitude','longitude','year','example_path','File_ID','version_no']) #create dataframe for this augmented image
    df_ev['example_path'] = 'rain_test_data/train/{p}_{q}.png'.format(p=df_ev['File_ID'].iloc[-1],q=int(df_ev['version_no'].iloc[-1])) #set the exmaple path in the dataframe
    df = pd.concat([df,df_ev],ignore_index=True) #combine the dataframe to the original one with all images in
    io.imsave('{h}/{p}_{q}.png'.format(h=outpath,k=folders[boolarr][0],p=df['File_ID'].iloc[-1],q=int(df['version_no'].iloc[-1])),arr=flipUD) #save the new augmented image
  return lengths, percentages, df #return the number of images in each scenario (lengths), their percentages, and the new dataframe with augmented images included.

#use function to augment images
lengths, percentages,df = percentage_matcher(30,df,df_0,df_1,df_2,path+'/train_test_data'+'/train',path+'/figs') 


#save the dataframe 
df.to_csv(path+'/new_train.csv')


print(len(df))

# visualise as pie chart
plt.figure(figsize = (8,8))
plt.pie(df.groupby('label').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

