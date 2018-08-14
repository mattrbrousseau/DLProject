
# coding: utf-8

# In[4]:


#path to where the folder "database" is
basepath = "/share/volatile_scratch/miria/SoundTagging/"

import numpy as np
import pandas as pd
import librosa
import simpleclassifiers_1 as clf
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

print("Matching target with classes")

for i in range(len(target)):
    
    if target[i] == "Computer_keyboard":
        target[i] = "Computer keyboard"
    if target[i] == "Microwave_oven":
        target[i] = "Microwave oven"
    if target[i] == "Keys_jangling":
        target[i] = "Keys jangling"
    if target[i] == "Acoustic_guitar":
        target[i] = "Acoustic guitar"
    if target[i] == "Gunshot_or_gunfire":
        target[i] = "Gunshot, gunfire"
    if target[i] == "Snare_drum":
        target[i] = "Snare drum"
    if target[i] == "Microwave_oven":
        target[i] = "Microwave oven"
    if target[i] == "Bass_drum":
        target[i] = "Bass drum"
    if target[i] == "Electric_piano":
        target[i] = "Electric piano"
    if target[i] == "Double_bass":
        target[i] = "Double bass"
    if target[i] == "Finger_snapping":
        target[i] = "Finger snapping"
    if target[i] == "Burping_or_eructation":
        target[i] = "Burping, eructation"
    if target[i] == "Drawer_open_or_close":
        target[i] = "Drawer open or close"
    if target[i] == "Violin_or_fiddle":
        target[i] = "Violin, fiddle"
    target[i].replace("_"," ")

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


clf.model_4hLayers(mfccPAD2600,target,"4hLayers",basepath)



