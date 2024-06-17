import sklearn
import pickle
import feature
import numpy as np
import dataset
def importModels(instrument):
    with open(f"Models/{instrument}.pkl", 'rb') as handle:
        model = pickle.load(handle)
    return model

def predict(audio_path):
    audio_array,_ = dataset.librosa.load(audio_path, sr=44100)
    windows = dataset.get_windows(audio_array)
    df = dataset.create_dataframe()
    df = dataset.append_windows("custom",windows,df)
    ampenv = feature.ampenv(df,sr=44100,audio_array=audio_array)
    rmse = feature.rmse(df,sr=44100,audio_array=audio_array)
    zcr =feature.zcr(df,sr=44100,audio_array=audio_array)
    sc = feature.sc(df,sr=44100,audio_array=audio_array)
    mfcc = feature.mfcc(df,sr=44100,audio_array=audio_array)
    lpc = feature.lpc(df,sr=44100,audio_array=audio_array)

    features = [ampenv,rmse,zcr,sc,mfcc,lpc]
    df = feature.merge(df,features)
    features_df = ['ampenv', 'rmse', 'zcr', 'spectral_centroid', 'mfcc_0', 'mfcc_1',
       'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'lpc_0', 'lpc_1', 'lpc_2',
       'lpc_3', 'lpc_4', 'lpc_5']
    X = df[features_df]
    violin_model = importModels("violin")
    y_violin = violin_model.predict(X)
    voice_model = importModels("voice")
    y_voice = voice_model.predict(X)
    mridangam_model = importModels("mridangam")
    y_mridangam = mridangam_model.predict(X)
    data = {
        "file_name": audio_path,
        "sr":44100,
        "audioLength": len(audio_array),
        "windowLength": len(y_violin),
        "audio_array": audio_array,
        "is_voice": y_voice,
        "is_violin":y_violin,
        "is_mridangam":y_mridangam
    }
    return data