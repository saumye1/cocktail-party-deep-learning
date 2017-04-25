import sys
import librosa
import numpy as np
import pickle


filename = sys.argv[1]
filename2 = sys.argv[2]

file = open("label_bool.data", "wb+")

fin = np.matrix(np.zeros((150, 300 * 260 ), dtype = bool))
padded1 = np.matrix(np.zeros((300, 260)))
padded2 = np.matrix(np.zeros((300, 260)))

for i in range(150):
	y1, sr1 = librosa.load(filename, offset = i * 6, duration = 6)
	y2, sr2 = librosa.load(filename2, offset = i * 6, duration = 6)
	mag_stft1 = np.abs(librosa.stft(y1))
	mag_stft2 = np.abs(librosa.stft(y2))
	cropped1 = mag_stft1[0:300, :]
	cropped2 = mag_stft2[0:300, :]
	padded1[:, 0:259] = cropped1
	padded2[:, 0:259] = cropped2
	reshaped1 = np.reshape(padded1, (1, 300 * 260))
	reshaped2 = np.reshape(padded2, (1, 300 * 260))

	#fin[i, :] = np.array([1 for j in range(300 * 259) if reshaped1[0][j] > reshaped2[0][j]])
	for j in range(300 * 260):
		if reshaped1[0, j] > reshaped2[0, j]:
			fin[i, j] = 1
		else:
			fin[i, j] = 0

	#plt.imshow(mag_stft, aspect = 'auto')
	#s  = "./images/image"
	#s += str(i) + ".png"
	#plt.savefig(s, bbox_inches = "tight")
pickle.dump(fin, file)

file.close()