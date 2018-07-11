
# Project: Freesound General-Purpose Audio Tagging

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

* Create audio descriptions [^1] => **I just saw we kind of have to use CNN =(**
[^1]: I think the easy way is to convert the signal into numbers instead of working with the image of the soundtrack or another approach. Once we have this working, we can improve the approch to something more fancy =)

    * Possible useful papers (They still need to be checked and we need to find more I think =/):
        - [Genre Classification using Independent and Discriminant Features] (https://jcis.sbrt.org.br/jcis/article/view/515/379)
        - [Audio Descriptors and Descriptor Schemes in the Context of MPEG-7] (http://www.imm.dtu.dk/~lfen/Audio%20Descriptors%20and%20Descriptor%20Schemes.pdf) This paper came from the same University that the author of the Kaggle competition are.
    
###At the Python Notebook (Project/Freesound General-Purpose Audio Tagging.ipynb):

* Load database using pandas => **We may have to use tensorflow => this info needs to be checked**

* Create classifier

* Improve classifier results

