import librosa
import audio_utils
import numpy as np

dir = "../datasets/IEMOCAP/All"

filenames = librosa.util.find_files(dir)

filename = filenames[0]

audio = audio_utils.load_wav(filename)
print(audio.shape)
audio = np.array(audio, dtype = np.float32)
mel = audio_utils.wav2melspectrogram(audio)

np.save('./samples/Test/mel.npy', mel)
