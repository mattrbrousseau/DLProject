
# Project: Freesound General-Purpose Audio Tagging

Description:
This task addresses the problem of general-purpose automatic audio tagging and poses two main challenges. The first one, which will be addressed in this project, is to build models that can recognize an increased number of sound events of very diverse nature, including musical instruments, human sounds, domestic sounds, animals, etc.

<link rel="stylesheet" type="text/css" href="http://dcase.community/challenge2018/task-general-purpose-audio-tagging"> Task entire description
<link rel="stylesheet" type="text/css" href="https://www.kaggle.com/c/freesound-audio-tagging"> Kaggle page
<link rel="stylesheet" type="text/css" href="https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-audio-data"> Beginners guide to audio data
<link rel="stylesheet" type="text/css" href="https://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis"> Analysis of the sound waves with visualizations

<b>Baseline solution description</b>
System description
The baseline system implements a convolutional neural network (CNN) classifier similar to, but scaled down from, the deep CNN models that have been very successful in the vision domain. The model takes framed examples of log mel spectrogram as input and produces ranked predictions over the 41 classes in the dataset. The baseline system also allows training a simpler fully connected multi-layer perceptron (MLP) classifier. The baseline system is built on TensorFlow.

<strong>Input features (audio description was used)</strong> 
We use frames of log mel spectrogram as input features:
    *computing spectrogram with a window size of 25ms and a hop size of 10ms
    *mapping the spectrogram to 64 mel bins covering the range 125-7500 Hz
    *log mel spectrogram is computed by applying log(mel spectrogram + 0.001)
    *log mel spectrogram is then framed into overlapping examples with a window size of 0.25s and a hop size of 0.125s

<strong>Architecture</strong>
The baseline CNN model consists of three 2-D convolutional layers (with ReLU activations) and alternating 2-D max-pool layers, followed by a final max-reduction (to produce a single value per feature map), and a softmax layer. The Adam optimizer is used to train the model, with a learning rate of 1e-4. A batch size of 64 is used.

The layers are listed in the table below using notation Conv2D(kernel size, stride, # feature maps) and MaxPool2D(kernel size, stride). Both Conv2D and MaxPool2D use the SAME padding scheme. ReduceMax applies a maximum-value reduction across the first two dimensions. Activation shapes do not include the batch dimension.

Layer   Activation shape
Input   (25, 64, 1)
Conv2D(7x7, 1, 100) (25, 64, 100)
MaxPool2D(3x3, 2x2) (13, 32, 100)
Conv2D(5x5, 1, 150) (13, 32, 150)
MaxPool2D(3x3, 2x2) (7, 16, 150)
Conv2D(3x3, 1, 200) (7, 16, 200)
ReduceMax   (1, 1, 200)
Softmax (41,)

<strong>Clip prediction</strong> 
The classifier predicts 41 scores for individual 0.25s-wide examples. In order to produce a ranked list of predicted classes for an entire clip, we average the predictions from all framed examples generated from the clip, and take the top 3 classes by score.

<strong>System performance</strong>
The baseline system trains to achieve an MAP@3 of ~0.7 on the public Kaggle leaderboard after ~5 epochs of the entire training set which are completed in ~12 hours on an n1-standard-8 Google Compute Engine machine with a quad-core Intel Xeon E5 v3 (Haswell) @ 2.3 GHz.

##Main tasks

* Check the rules' competition and input a summary with requirements and limitations

* Download database - **Done-2018/06/27**
    * Project/database/

* Database description - **Done-2018/06/27**
    * Train set
        - It is in Audio_train folder.
        - wav files
        - Includes ~9.5k samples unequally distributed among 41 categories.
        - Minimum number of audio samples per category is 94, and the maximum 300. 
        - Duration of the audio samples ranges from 300ms to 30s.
        - Composed of ~3.7k manually-verified annotations and ~5.8k non-verified annotations. The quality of the non-verified annotations is of at least 65-70% in each sound category.

    * Train set labels
        - It is in the train.csv file, with format: <file_name>,<label>
        
    * Test set
        - It is in Audio_test folder.
        - wav files
        - Includes 9.4k samples
    
    * Test set labels
        - Find the labels

    * labels = ["Tearing","Bus","Shatter","Gunshot, gunfire","Fireworks","Writing",\
        "Computer keyboard","Scissors","Microwave oven","Keys jangling",\
        "Drawer open or close","Squeak","Knock","Telephone","Saxophone",\
        "Oboe","Flute","Clarinet","Acoustic guitar","Tambourine","Glockenspiel",\
        "Gong","Snare drum","Bass drum","Hi-hat","Electric piano","Harmonica",\
        "Trumpet","Violin, fiddle","Double bass","Cello","Chime","Cough",\
        "Laughter","Applause","Finger snapping","Fart","Burping, eructation",\
        "Cowbell","Bark","Meow"]

* Create audio descriptions [^1] => **Use librosa to extracte features and use https://www.kaggle.com/codename007/a-very-extensive-freesound-exploratory-analysis what is each feature about**
[^1]: I think the easy way is to convert the signal into numbers instead of working with the image of the soundtrack or another approach. Once we have this working, we can improve the approch to something more fancy =)

    * Possible useful papers (They still need to be checked and we need to find more I think =/):
        - [Genre Classification using Independent and Discriminant Features] (https://jcis.sbrt.org.br/jcis/article/view/515/379)
        - [Audio Descriptors and Descriptor Schemes in the Context of MPEG-7] (http://www.imm.dtu.dk/~lfen/Audio%20Descriptors%20and%20Descriptor%20Schemes.pdf) This paper came from the same University that the author of the Kaggle competition are.
    
###At the Python Notebook (Project/Freesound General-Purpose Audio Tagging.ipynb):

* Load database using pandas => **We may have to use tensorflow => this info needs to be checked**

* Create classifier

* Improve classifier results

