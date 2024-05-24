import dataset
dataset.initDataset(r"C:\Users\w190973\OneDrive - Worldline SA\Escritorio\mtg\dataset")
song = dataset.get_random_song()
print(dataset.get_tonic(song.track_id))