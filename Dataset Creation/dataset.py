import mirdata
import pandas as pd
import librosa 
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from math import ceil

window_sec = 1

def init_dataset(path):
    """
    Initialize the dataset from the path
    """
    global saraga
    saraga = mirdata.initialize("saraga_carnatic",data_home=path)

def get_metadata(track_id):
    """
    For <track_id>, return a dataframe of associated metadata
    """
    metadata = saraga.track(track_id).metadata
    
    return metadata

def get_performer(track_id):
    """
    For <track_id>, return the performer
    """
    performer = saraga.track(track_id).metadata["album_artists"][0]["name"]
    return performer


def get_tonic(track_id):
    """
    For <track_id>, return the tonic in hertz
    """
    tonic = saraga.track(track_id).tonic
    return tonic

def get_random_song():
    """
    Return a random song from the dataset
    """
    song = saraga.choice_track()
    return song

def load_mixed_audio(track_id):
    """
    For <track_id>, return the loaded audio
    """
    audio_path = saraga.track(track_id).audio_path
    audio_array, sr = librosa.load(audio_path, sr=44100)    
    return audio_array

def load_violin_audio(track_id):
    """
    For <track_id>, return the isolated violin track
    """
    audio_path = saraga.track(track_id).audio_violin_path
    audio_array, sr = librosa.load(audio_path, sr=44100)
    return audio_array

def load_voice_audio(track_id):
    """
    For <track_id>, return the isolated voice track
    """
    audio_path = saraga.track(track_id).audio_vocal_path
    if audio_path != None:
        audio_array, sr = librosa.load(audio_path, sr=44100)    
        return audio_array
    else:
        return None

def load_mridangam_audio(track_id):
    """
    For <track_id>, return the isolated mridangam track
    """
    audio_path_right = saraga.track(track_id).audio_mridangam_right_path
    audio_path_left = saraga.track(track_id).audio_mridangam_left_path

    audio_array_right, sr = librosa.load(audio_path_right, sr=44100,mono=False)    
    audio_array_left, sr = librosa.load(audio_path_left, sr=44100,mono=False)
    audio_array = librosa.to_mono(np.array([audio_array_left,audio_array_right]))
    return audio_array

def plot_waveform(audio_array,amplitude="normal"):
    """
    Plot waveform for <audio_array> using matplotlib.pyplot
    """
    
    if amplitude=="dB":
        audio_array = 20*np.log10(audio_array)
    plt.figure().set_figwidth(20)
    minutes = np.arange(len(audio_array))/(44100*60)
    # plt.magnitude_spectrum(audio_array, Fs=44100, scale='dB', color='C1')
    plt.plot(minutes,audio_array,zorder=0)
    plt.xlabel('Minutes')
    plt.ylabel('Amplitude')
    plt.show()


def play_audio(audio_array):
    """
    Generate audio player for <audio_array> using Ipython library
    """
    Audio(audio_array, rate=44100)

def detect_silence(audio_array,th=20):
    """
    Return array of 0 and 1 (is silent/is not silent) for input <audio_array>. Returned array should
    be equal in length to input array
    """
    silent = librosa.effects.split(audio_array, top_db=th)
    size = len(audio_array)
    is_silent = np.zeros(size)
    for i in silent:
        is_silent[i[0]:i[1]] = 1
    return is_silent
def plot_silence(is_silent,audio_array):
    plt.figure().set_figwidth(20)
    samples =  [(x/44100)/60 for x in range(len(audio_array))]
    plt.plot(samples,audio_array,zorder=0)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Amplitude')
    silent_samples = ((np.where(is_silent==0)[0])/44100)/60
    plt.scatter(silent_samples,np.zeros(len(silent_samples)),color='red',zorder=1,s=0.1)
    plt.show()

def get_multitrack_songs():
    trackIds = []
    for i in saraga.track_ids:
        if saraga.track(i).audio_vocal_path is not None:
            trackIds.append(i)
    return trackIds
def get_windows(audio_array):
    w_samples = window_sec*44100

    num_windows = ceil(len(audio_array)/w_samples)

    windows = np.array_split(audio_array, num_windows) 
    return windows
def get_target(silence_windows,th=0.5):
    num_windows = len(silence_windows)
    target = np.zeros(num_windows)
    for i,arr in enumerate(silence_windows):
        ones = np.sum(arr==1)
        perc= ones/len(arr)
        if perc > th:
            target[i] = 1
    return target
def append_dataframe(track_id,violin_target,voice_target,mridangam_target,df):
    for i in range(len(violin_target)):
        row = {
            "track_id": track_id,
            "window_index":i,
            "is_violin": violin_target[i],
            "is_voice": voice_target[i],
            "is_mridangam": mridangam_target[i]
        }
        df = df.append(row,ignore_index=True)
    return df
def create_dataframe():
    df = pd.DataFrame(columns=["track_id","window_index","is_violin","is_voice","is_mridangam"])
    return df
def plot_instrument(audio_array,instrument_target):
    samples_w = window_sec*44100
    print(len(audio_array))
    print(samples_w*len(instrument_target))
    plt.figure().set_figwidth(20)
    samples =  [(x/44100)/60 for x in range(len(audio_array))]
    plt.plot(samples,audio_array,zorder=0)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Amplitude')
    # silent_samples = ((np.where(is_silent==0)[0])/44100)/60
    # plt.scatter(silent_samples,np.zeros(len(silent_samples)),color='red',zorder=1,s=0.1)
    # plt.show()
    
    silent_samples_w = np.where(instrument_target==1)[0]
    for i in range(len)):

        if instrument_target[i] == 1:
            

  
