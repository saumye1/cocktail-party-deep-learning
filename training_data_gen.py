import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np


filename = sys.argv[1]

for i in range(0. 150):
	y, sr = librosa.load(filename, offset = i * 6, duration = 6)
	mag_stft = np.abs(librosa.stft(y))
	plt.imshow(mag_stft, aspect = 'auto')
	s  = "./images/image"
	s += str(i) + ".png"
	plt.savefig(s, bbox_inches = "tight")
