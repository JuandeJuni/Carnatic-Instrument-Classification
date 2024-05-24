import mirdata
import pandas as pd

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