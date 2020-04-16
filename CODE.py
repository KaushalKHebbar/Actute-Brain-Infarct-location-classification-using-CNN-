#!/usr/bin/env python
# coding: utf-8

# In[130]:



#importing necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Input, ZeroPadding2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from keras.models import Model, load_model
from keras import losses 
from keras import optimizers 
from keras import metrics 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from os import listdir
import PIL
from sklearn.utils import shuffle
import cv2
#import imutils
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from os import listdir
import glob

#data_gen = ImageDataGenerator(    rotation_range=10, 
#                                  width_shift_range=0.05, 
#                                  height_shift_range=0.05, 
#                                  shear_range=0.1, 
#                                  fill_mode='nearest',
#                                  interpolation_order=5
#                             )

#augmented_image_paths=glob.glob(r'C:\Users\sonuk\OneDrive\Desktop\DWI_DATA\DWI_train_u\**\*DWI.jpg', recursive=True)
#number_of_augs=5
#for name in augmented_image_paths:
#    image_path=name
#    image=cv2.imread(image_path)
#    image = image.reshape((1,)+image.shape)
#    save_image_as = 'Aug_'+image_path.split('\\')[-1][:-4]
#    save_folder_as=image_path.split('\Case')[0]
#    i=1
#    for batch in data_gen.flow(x=image, batch_size=1, save_to_dir=save_folder_as, save_prefix=save_image_as, save_format='jpg'):
#        i += 1
#        if i>number_of_augs:
#            break

#to get the list of labels i.e. infarct location folader names
def labels_list(image_path_list):
    label_list=[]
    for image_path in image_path_list:
        label=image_path.split('\\')[-2]
        label_list.append(label)
    return label_list



#to return a nuemeric label i.e. index of it in the folders list
def label_map(image_path,label_name_list):
    label_name=image_path.split('\\')[-2]
    return label_name_list.index(label_name)


def find_index(image_path):
    i=image_path.split('\\')[-2]
    
    if i=='Bilateral cerebellar hemispheres': 
        return 0
    if i=='Bilateral frontal lobes':
        return 1
    if i=='Bilateral occipital lobes':
        return 2
    if i=='Brainstem':
        return 3
    if i=='Dorsal aspect of pons': 
        return 4
    if i=='Left centrum semi ovale and right parietal lobe':
        return 5
    if i=='Left cerebellar':
        return 6
    if i=='Left corona radiata':    
        return 7
    if i=='Left frontal lobe':  
        return 8
    if i=='Left frontal lobe in precentral gyral location':
        return 9
    if i=='Left Fronto parietal':
        return 10
    if i=='Left insula':
        return 11
    if i=='Left occipital and temporal lobes':
        return 12
    if i=='Left occipital lobe':
        return 13
    if i=='Left parietal lobe':
        return 14
    if i=='Left thalamic':
        return 15
    if i=='Medial part of right frontal and parietal lobes':
        return 16
    if i=='Medula oblongata-left':
        return 17
    if i=='Mid brain on right side':
        return 18
    if i=='Pons-left':
        return 19
    if i=='Pontine-right':
        return 20
    if i=='posterior limb of left internal capsule':
        return 21
    if i=='Right anterior thalamic': 
        return 22
    if i=='Right cerebellar hemisphere': 
        return 23
    if i=='Right corona radiata':
        return 24
    if i=='Right frontal lobe':  
        return 25
    if i=='Right fronto-parieto-temporo- occipital lobes':      
        return 26
    if i=='Right ganglio-capsular region':
        return 27
    if i=='Right insula': 
        return 28
    if i=='Right lentiform nucleus':   
        return 29
    if i=='Right occipital lobe':
        return 30
    if i=='Right parietal lobe':   
        return 31
    if i=='Right putamen':  
        return 32
    if i=='Right temporal lobe': 
        return 33
    if i=='Right thalamus':
        return 34
    if i=='Splenium of the corpus callosum': 
        return 35




def load_data(image_path_list, image_size, label_name_list):
    X = []
    y = []
    image_width, image_height = image_size
    
    for image_path in image_path_list:
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)
        image = image / 255.
        X.append(image)
        #returns the index==label number of brain location folders
        #label_num=label_map(image_path,label_name_list)
        index=find_index(image_path)
        y.append(index)    
        
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle the data
    X, y = shuffle(X, y)
    
    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')
    #print(y)

    return X, y

train_image_paths_list=glob.glob(r'C:\Users\sonuk\OneDrive\Desktop\DWI_DATA\Combined_DATA_train_cpy\**\*.jpg', recursive=True)
test_image_paths_list=glob.glob(r'C:\Users\sonuk\OneDrive\Desktop\DWI_DATA\DWI_test\**\*.jpg', recursive=True)

#returns list of label names of brain location folders i.e. train_image_paths_list=all the possible labels
label_name_list=labels_list(train_image_paths_list)

IMG_WIDTH, IMG_HEIGHT = (128, 128)

X_train, y_train = load_data(train_image_paths_list, (IMG_WIDTH, IMG_HEIGHT), label_name_list)
X_test, y_test = load_data(test_image_paths_list, (IMG_WIDTH, IMG_HEIGHT), label_name_list)



# In[157]:


def try_model(X_train,y_train):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(5,5), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,3)))
    #model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    #model.add(Conv2D(32, kernel_size=(3,3), activation='relu')) 
    #model.add(Dropout(0,5))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0,5))
    #model.add(Dense(92))
    #model.add(Activation('relu'))
    model.add(Dense(36))
    model.add(Activation('softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    #model.fit(X_train, y_train, batch_size =32,validation_data=(X_val,y_val),epochs=8)
    model.fit(X_train, y_train, batch_size =32,validation_split=0.3,epochs=8)
    return model




mdl=try_model(X_train,y_train)#,X_val,y_val)
print(mdl.summary())



# In[150]:


history = mdl.history.history

def plot_metrics(history):
    
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']
    
    # Loss
    plt.subplot(121)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    
    # Accuracy
    plt.subplot(122)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

plot_metrics(history) 




# In[158]:


result = mdl.predict(X_test)
print('\n')
t0=np.argmax(result[0])
t1=np.argmax(result[1])
t2=np.argmax(result[2])
t3=np.argmax(result[3])
t4=np.argmax(result[4])
print('The predicted output is:')
print(t0,t1,t2,t3,t4)


print('The index must be: ')
print(list(y_test))

directory = r'C:\Users\sonuk\OneDrive\Desktop\DWI_DATA\Combined_DATA_train'
y_path= os.listdir(directory)

#print(y_path[t0]+'\n'+y_path[t1]+'\n'+y_path[t2]+'\n'+y_path[t3]+'\n'+y_path[t4])
#print(y_path)


# In[138]:


list(t0,t1)


# In[86]:


im=cv2.imread(r'C:\Users\sonuk\OneDrive\Desktop\DWI_DATA\DWI_test_temp\Bilateral cerebellar hemispheres\Case 12_DWI.jpg')
im = cv2.resize(im, dsize=(128,128), interpolation=cv2.INTER_CUBIC)


# In[87]:


im/255.


# In[ ]:




