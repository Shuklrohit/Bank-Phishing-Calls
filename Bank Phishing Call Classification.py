#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io.wavfile import read
from matplotlib.pyplot import specgram
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


pip install pydub


# In[5]:


pip install ffmpeg


# In[6]:


from os import path
from pydub import AudioSegment


# In[36]:


import IPython.display as ipd  # To play sound in the notebook
fname = r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\wavformat/' + 'Fcall1.wav'   # Hi-hat
ipd.Audio(fname)


# In[63]:


import wave
wav = wave.open(fname)
print("Sampling (frame) rate = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate())


# In[117]:


import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper/SpamCall Claassification.csv')

import os
import struct
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import IPython.display as ipd

def path_class(filename):
    excerpt = data[data['Name'] == filename]
    path_name = os.path.join(r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\waveformat', filename)
    return path_name, excerpt['class'].values[0]
  
def wav_fmt_parser(file_name):
    full_path, _ = path_class(file_name)
    wave_file = open(full_path,"rb")
    riff_fmt = wave_file.read(36)
    n_channels_string = riff_fmt[22:24]
    n_channels = struct.unpack("H",n_channels_string)[0]
    s_rate_string = riff_fmt[24:28]
    s_rate = struct.unpack("I",s_rate_string)[0]
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    return (n_channels,s_rate,bit_depth)


# In[118]:


def wav_plotter(full_path, class_label):   
    rate, wav_sample = wav.read(full_path)
    wave_file = open(full_path,"rb")
    riff_fmt = wave_file.read(36)
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    print('sampling rate: ',rate,'Hz')
    print('bit depth: ',bit_depth)
    print('number of channels: ',wav_sample.shape[1])
    print('duration: ',wav_sample.shape[0]/rate,' second')
    print('number of samples: ',len(wav_sample))
    print('class: ',class_label)
    plt.figure(figsize=(12, 4))
    plt.plot(wav_sample) 
    return ipd.Audio(full_path)


# In[119]:


# wav_fmt_data = [wav_fmt_parser(i) for i in data.name]
# data[['n_channels','sampling_rate','bit_depth']] = pd.DataFrame(wav_fmt_data)
# data.head'()
# fullpath, label = path_class('0.wav')
# wav_plotter(fullpath,label)
data


# In[ ]:


from scipy.io import wavfile
import matplotlib.pyplot as plt 
import numpy as np
import wave
filepath = r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\wavformat/'
filename = 0

while filename <= 7:
    rate, data = wavfile.read(filepath +  str(filename )+'.wav')
    print("Sampling (frame) rate = ", rate)
    print("Total samples (frames) = ", data.shape)
    print(data)
    plt.plot(data.T )
    plt.show()
    filename +=1


# In[123]:


def wav_plotter(full_path, class_label):   
    rate, wav_sample = wav.read(full_path)
    wave_file = open(full_path,"rb")
    riff_fmt = wave_file.read(36)
    bit_depth_string = riff_fmt[-2:]
    bit_depth = struct.unpack("H",bit_depth_string)[0]
    print('sampling rate: ',rate,'Hz')
    print('bit depth: ',bit_depth)
    print('number of channels: ',wav_sample.shape[1])
    print('duration: ',wav_sample.shape[0]/rate,' second')
    print('number of samples: ',len(wav_sample))
    print('class: ',class_label)
    plt.figure(figsize=(12, 4))
    plt.plot(wav_sample) 
    return ipd.Audio(full_path)


# In[ ]:





# In[124]:


def path_class(filename):
#     excerpt = data[data['slice_file_name'] == filename]
    path_name = r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\wavformat/' + filename + '.wav'
    return path_name


# In[84]:


path_class('1')


# In[125]:


fullpath = path_class('0')
wav_plotter(fullpath,0)


# In[4]:


# fullpath = path_class('Audio-1')
# wav_plotter(fullpath,1)


# In[1]:


# fullpath = path_class('3')
# wav_plotter(fullpath,1)


# In[2]:


# fullpath = path_class('2')
# wav_plotter(fullpath,1)


# In[142]:


def extract_features(file_name):
   
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print("Error encountered while parsing file: ", file)
        return None 
     
    return mfccsscaled

# Load various imports 
# import pandas as pd
# import os
# import librosa

# # Set the path to the full UrbanSound dataset 
# fulldatasetpath = r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\wavformat/'

# metadata = pd.read_csv(r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper/SpamCall Claassification.csv')

# features = []

# # Iterate through each sound file and extract the features 
# for index, row in metadata.iterrows():
    
#     file_name = os.path.join(os.path.join(r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\waveformat',str(row['Name']))
#     class1 = row['class_name']
#     data = extract_features(file_name)
    
#     features.append([data, class_label])

# # Convert into a Panda dataframe 
# featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

# print('Finished feature extraction from ', len(featuresdf), ' files')


# In[2]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import numpy as np
def create_fold_spectrograms(fold):
    spectrogram_path =  r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\Spactograph/'  
    audio_path =  r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\wavformat/'
    print(f'Processing fold {fold}')
    os.mkdir(spectrogram_path/fold)
    for audio_file in list(Path(audio_path/f'fold{fold}').glob('*.wav')):
        samples, sample_rate = librosa.load(audio_file)
        fig = plt.figure(figsize=[0.72,0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        filename  = spectrogram_path/fold/Path(audio_file).name.replace('.wav','.png')
        S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')


# # building model

# In[17]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tqdm import tqdm


# In[30]:


train = pd.read_csv(r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\images/Train.csv')


# In[31]:


# We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False
train_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img(r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\images/'+train['Name'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)


# In[33]:


y=train['CallType'].values
y = to_categorical(y)


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


# In[38]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))


# In[39]:


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])


# In[40]:


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# In[41]:


test = pd.read_csv(r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\images/test.csv')


# In[43]:


test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img(r'C:\Users\rohit.c.shukla\Desktop\ML\WhitePaper\images/'+test['Name'][i].astype('str')+'.png', target_size=(28,28,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
test = np.array(test_image)


# In[44]:


# making predictions
prediction = model.predict_classes(test)


# In[47]:


score = model.evaluate(X_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

