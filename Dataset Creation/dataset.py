import mirdata
import pandas as pd
import librosa 
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio

def initDataset(path):
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
    audio_array = librosa.to_mono(np.array([audio_path_left,audio_path_right]))
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
    plt.plot(minutes,audio_array)
    plt.xlabel('Minutes')
    plt.ylabel('Amplitude')
    plt.show()


def play_audio(audio_array):
    """
    Generate audio player for <audio_array> using Ipython library
    """
    display(Audio(audio_array, rate=44100))

