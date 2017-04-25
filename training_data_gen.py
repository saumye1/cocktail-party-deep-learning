import sys
import librosa
import numpy as np
import pickle


filename = sys.argv[1]
filename2 = sys.argv[2]

file = open("train.data", "wb+")

fin = np.matrix(np.zeros((150, 300 * 259 )))

for i in range(150):
	y1, sr1 = librosa.load(filename, offset = i * 6, duration = 6)
	y2, sr2 = librosa.load(filename2, offset = i * 6, duration = 6)
	mag_stft = np.abs(librosa.stft(y1 + y2))
	cropped = mag_stft[0:300, :]

	fin[i, :] = np.reshape(cropped, (1, 300 * 259))

	#plt.imshow(mag_stft, aspect = 'auto')
	#s  = "./images/image"
	#s += str(i) + ".png"
	#plt.savefig(s, bbox_inches = "tight")
pickle.dump(fin, file)

file.close()
