''' extract_mfbs 

Extract Mel-filterbanks (MFBs) 

Reference: 
https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

MADDoG Paper:
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8713918


'''

import argparse 
import numpy as np
import librosa
import os 
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt 

#Parameters
#***********************
#sample rate (Hz) 
#NOTE: PRIORI Data sampled at 8000 Hz 
_SAMPLE_RATE = 16000
#Number of FFTs
_N_FFT = 2048
#Window length 
_WIN_LENGTH = 200
#Hop length 
_HOP_LENGTH = 80
#Minimum frequency (Hz) 
_FMIN = 0
#Maximum frequency (Hz) 
_FMAX = None
#Number of mel-filters 
_N_MELS = 40


def extract_mfb(y, output_file):
    ''' Extract MFBs using the librosa package 

    Inputs:
        y - audio data 
        output_file - full output file path to save MFB as numpy file
    Outputs: 
        None
    ''' 

    #Pre-emphasis (amplifies high frequencies) 
    y = librosa.effects.preemphasis(np.array(y))

    #Apply short-time fourier transform 
    spec = librosa.core.stft(y=y,
                             n_fft=_N_FFT,
                             hop_length=_HOP_LENGTH,
                             win_length=_WIN_LENGTH,
                             window='hann',
                             center=True,
                             pad_mode='reflect')
    #Get the magnitude of the spectrogram 
    spec = librosa.magphase(spec)[0]
    #Compute the melspectrogram 
    mel_spectrogram = librosa.feature.melspectrogram(S=spec,
                                                     sr=_SAMPLE_RATE,
                                                     n_mels=_N_MELS,
                                                     power=1.0,  # actually not used given "S=spec"
                                                     fmin=_FMIN,
                                                     fmax=_FMAX,
                                                     htk=False,
                                                     norm=1)
    
    #Take the log of melspectrogram (add small value for numerical stability) 
    log_mel_spectrogram = np.log(mel_spectrogram + 1e-6).astype(np.float32)
        
    #z-normalize utterance 
    stdVal = np.std(log_mel_spectrogram) 
    if stdVal == 0: 
        stdVal = 1 
    log_mel_spectrogram = (log_mel_spectrogram - np.mean(log_mel_spectrogram)) / stdVal 

    # clamp to 3 std (removes outliers in MFB - critical for MADDoG performance)
    clampVal = 3.0 
    log_mel_spectrogram[log_mel_spectrogram > clampVal] = clampVal 
    log_mel_spectrogram[log_mel_spectrogram < -clampVal] = -clampVal 

    #save transposed version of MFB (compatible with MADDoG architecture) 
    np.save(output_file, arr=np.transpose(log_mel_spectrogram))



def main():
    #load an array of audio id's
    data=np.load("./segment_id.npy")
    _SAMPLE_RATE = 16000
    for id in data:
    	path=os.path.join("./wav",(str(id)+".wav"))
    	data, rate = librosa.core.load(path=path, sr=_SAMPLE_RATE)
    	output_file = os.path.join("/home/lancelcy/Priori/mfbs", str(id)+'.npy')
    	extract_mfb(data, output_file)
    
if __name__ == "__main__":
    main()
     