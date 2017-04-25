import librosa
import matplotlib.pyplot as plt
import numpy as np
import sys

filename = sys.argv[1]

y, sr = librosa.load(filename)
D = np.abs(librosa.stft(y))

plt.imshow(D, aspect = 'auto')
plt.savefig("test2.png", bbox_inches = "tight")