import dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

with open("dataset.csv") as f:
    df = pd.read_csv(f)
canso1df = df[df["track_id"] == "0_Dorakuna"]
dataset.init_dataset(path=r'C:\UPF\2023\3rd Term\Taller\DataAPI')
violin = dataset.load_violin_audio("0_Dorakuna")
violin_target = canso1df["is_violin"]
print(np.where(violin_target == 1)[0])
# dataset.plot_instrument(violin,violin_target)



