import librosa
import dataset
import pandas as pd

def mfcc(df,sr=44100,n_mfcc=2):
    tracks = df["track_id"].unique()
    feature_list = []

    for i in tracks:
        
        song_df = df[df["track_id"] == i].copy()
        
        # print(song_df)
        mixed_audio = dataset.load_mixed_audio(i)
        hop_length = dataset.window_sec*sr
        mfcc = librosa.feature.mfcc(y=mixed_audio,sr=sr,n_mfcc=n_mfcc,hop_length=hop_length,center=False)
        mfcc_df = pd.DataFrame(mfcc.T, columns=[f"mfcc_{c}" for c in range(n_mfcc)])
        
        mfcc_df["track_id"] = i
        mfcc_df["window_index"] = range(len(mfcc_df))
        
        merged_df = pd.merge(song_df, mfcc_df, on=["track_id", "window_index"], how="outer")
        
        feature_list.append(merged_df)
        break

    feature_dataframe = pd.concat(feature_list, ignore_index=True)
    print(feature_dataframe)
    return feature_dataframe

def rmse (df,sr=44100):
    tracks = df["track_id"].unique()
    feature_list = []

    for i in tracks:
        
        song_df = df[df["track_id"] == i].copy()
        
        # print(song_df)
        mixed_audio = dataset.load_mixed_audio(i)
        hop_length = dataset.window_sec*sr
        rmse = librosa.feature.rms(y=mixed_audio,frame_length= 2048, hop_length=hop_length,center=False)
        rmse_df = pd.DataFrame(rmse.T, columns=["rmse"])
        print(rmse_df)
        rmse_df["track_id"] = i
        rmse_df["window_index"] = range(len(rmse_df))
        
        merged_df = pd.merge(song_df, rmse_df, on=["track_id", "window_index"], how="outer")
        
        feature_list.append(merged_df)
        break

    feature_dataframe = pd.concat(feature_list, ignore_index=True)
    print(feature_dataframe)
    return feature_dataframe

def sc(df,sr=44100):
    tracks = df["track_id"].unique()
    feature_list = []

    for i in tracks:
        
        song_df = df[df["track_id"] == i].copy()
        
        # print(song_df)
        mixed_audio = dataset.load_mixed_audio(i)
        hop_length = dataset.window_sec*sr
        spectral_centroid = librosa.feature.spectral_centroid(y=mixed_audio,sr=sr,n_fft = 2048, hop_length=hop_length,center=False)
        spectral_centroid_df = pd.DataFrame(spectral_centroid.T, columns=["spectral_centroid"])
        
        spectral_centroid_df["track_id"] = i
        spectral_centroid_df["window_index"] = range(len(spectral_centroid_df))
        
        merged_df = pd.merge(song_df, spectral_centroid_df, on=["track_id", "window_index"], how="outer")
        
        feature_list.append(merged_df)
        break

    feature_dataframe = pd.concat(feature_list, ignore_index=True)
    print(feature_dataframe)
    return feature_dataframe

def lpc(df,sr=44100):
    tracks = df["track_id"].unique()
    feature_list = []

    for i in tracks:
        
        song_df = df[df["track_id"] == i].copy()
        
        # print(song_df)
        mixed_audio = dataset.load_mixed_audio(i)
        windows = dataset.get_windows(mixed_audio)
        lpc_df = pd.DataFrame(columns=[f"lpc_{c}" for c in range(6)])
        for w in windows:
            lpc = librosa.lpc(y=w,order=6)
            lpc_df.loc[len(lpc_df)] = lpc[1:]           
        print(lpc_df)
        lpc_df["track_id"] = i
        lpc_df["window_index"] = range(len(lpc_df))
        
        merged_df = pd.merge(song_df, lpc_df, on=["track_id", "window_index"], how="outer")
        
        feature_list.append(merged_df)
        break
    feature_dataframe = pd.concat(feature_list, ignore_index=True)
    print(feature_dataframe)
    return feature_dataframe