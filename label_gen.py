import sys
import librosa
#import matplotlib.pyplot as plt
import numpy as np
import pickle


filename = sys.argv[1]
filename2 = sys.argv[2]

file = open("label_bool.data", "wb+")

fin = np.matrix(np.zeros((150, 300 * 259 ), dtype = bool))

for i in range(150):
	y1, sr1 = librosa.load(filename, offset = i * 6, duration = 6)
	y2, sr2 = librosa.load(filename2, offset = i * 6, duration = 6)
	mag_stft1 = np.abs(librosa.stft(y1))
	mag_stft2 = np.abs(librosa.stft(y2))
	cropped1 = mag_stft1[0:300, :]
	cropped2 = mag_stft2[0:300, :]
	reshaped1 = np.reshape(cropped1, (1, 300 * 259))
	reshaped2 = np.reshape(cropped2, (1, 300 * 259))

	#fin[i, :] = np.array([1 for j in range(300 * 259) if reshaped1[0][j] > reshaped2[0][j]])
	for j in range(300 * 259):
		if reshaped1[0][j] > reshaped2[0][j]:
			fin[i, j] = 1
		else:
			fin[i, j] = 0

	#plt.imshow(mag_stft, aspect = 'auto')
	#s  = "./images/image"
	#s += str(i) + ".png"
	#plt.savefig(s, bbox_inches = "tight")
pickle.dump(fin, file)

file.close()