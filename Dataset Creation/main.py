import dataset
import librosa

dataset.initDataset(r"C:\Users\w190973\OneDrive - Worldline SA\Escritorio\mtg\dataset")
song = dataset.get_random_song()
print(dataset.get_tonic(song.track_id))

audio_path = r"C:\Users\w190973\OneDrive - Worldline SA\Escritorio\mtg\dataset"
audio_array, sr = librosa.load(audio_path, sr=44100)
print(dataset.play_audio(audio_array))
