import librosa
import dataset
import pandas as pd
def mfcc(df,sr=44100,n_mfcc=2):
    dataset.init_dataset(r"C:\UPF\2023\3rd Term\Taller\DataAPI")    
    tracks = df["track_id"].unique()
    feature_list = []

    for i in tracks:
        
        song_df = df[df["track_id"] == i].copy()
        
        # print(song_df)
        mixed_audio = dataset.load_mixed_audio(i)
        print(song_df.shape)
        hop_length = dataset.window_sec*sr
        mfcc = librosa.feature.mfcc(y=mixed_audio,sr=sr,n_mfcc=n_mfcc,hop_length=hop_length,center=False)
        mfcc_df = pd.DataFrame(mfcc.T, columns=[f"mfcc_{c}" for c in range(n_mfcc)])
        
        mfcc_df["track_id"] = i
        mfcc_df["window_index"] = range(len(mfcc_df))
        
        merged_df = pd.merge(song_df, mfcc_df, on=["track_id", "window_index"], how="outer")
        
        feature_list.append(merged_df)

    feature_dataframe = pd.concat(feature_list, ignore_index=True)
    print(feature_dataframe)
    return feature_dataframe
        


