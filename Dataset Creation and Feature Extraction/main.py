import dataset

dataset_path = r"C:\UPF\2023\3rd Term\Taller\DataAPI"
dataset.init_dataset(dataset_path)
track_ids = dataset.get_multitrack_songs()
df = dataset.create_dataframe()
for t in track_ids:
    w_size_sec = 1

    voice_track = dataset.load_voice_audio(t)
    voice_silence = dataset.detect_silence(voice_track)
    w_voice_silence = dataset.get_windows(voice_silence,w_size_sec)

    violin_track = dataset.load_violin_audio(t)
    violin_silence = dataset.detect_silence(violin_track,th=13)
    w_violin_silence = dataset.get_windows(violin_silence,w_size_sec)

    mridangam_track = dataset.load_mridangam_audio(t)
    mridangam_silence = dataset.detect_silence(mridangam_track)
    w_mridangam_silence = dataset.get_windows(mridangam_silence,w_size_sec)

    
    voice_target = dataset.get_target(w_voice_silence)
    violin_target = dataset.get_target(w_violin_silence)
    mridangam_target = dataset.get_target(w_mridangam_silence)
    
    df = dataset.append_dataframe(t,violin_target,voice_target,mridangam_target,df)

df.to_csv("dataset.csv",index=False)


