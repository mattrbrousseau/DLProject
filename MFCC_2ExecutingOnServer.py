
# coding: utf-8

# In[4]:


#path to where the folder "database" is
basepath = "/share/volatile_scratch/miria/SoundTagging/"

import numpy as np
import pandas as pd
import librosa
import simpleclassifiers as clf
'''
Piece of code copied from https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data
'''
np.random.seed(1001)
import os
import shutil

from scipy.io import wavfile

dataset = pd.read_csv(basepath+"database/train.csv",header=0,names=['fname','label','verified'])

checkedDataset = dataset.loc[dataset.verified == 1,["fname","label"]]

basepathWAVfiles = basepath+"database/audio_train/"

print("Loading Original dataset")
originalData = []
target = []
for sample in checkedDataset.values:
    fname,label = sample
    try:
        _,data = wavfile.read(basepathWAVfiles+fname)
        originalData.append(data.astype(np.float))
        target.append(label)
    except:
        print("File "+fname+' do not exist!') 
print("Calculating the MFCC")
mfcc = list(map(lambda x: librosa.feature.mfcc(y=x, sr=44100), originalData))

def normalize(v):
    '''
    Source of this functions: https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
    '''
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def paddingDescriptions(description,maxSize):
    '''
    Pads the descriptions into size maxSize and makes it 1D for each sample
    '''
    paddedData = []
    for d in description:
        newd=[]
        for i in d:
            i = np.pad(i,(0,maxSize-len(i)), 'constant')
            newd.append(i)
        d = np.reshape(np.array(newd),[-1])
        d = normalize(d)
        d = d.astype(np.float64)
        paddedData.append(d)
    return paddedData

print("Making all feature vector to have the same size")
mfccPAD2600 = paddingDescriptions(mfcc,2600)


clf.model_14hLayers(mfccPAD2600,target,"14hLayersAndMoreHUnits",basepath)



