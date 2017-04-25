import librosa
import matplotlib.pyplot as plt
import numpy as np
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

y1, sr1 = librosa.load(file1, duration = 6)
y2, sr2 = librosa.load(file2, duration = 6)

y = y1 + y2

D = np.abs(librosa.stft(y))
E = D[0:300, :]
plt.imshow(E, aspect = 'auto')
plt.savefig("whistle_guitar.png", bbox_inches = "tight")
