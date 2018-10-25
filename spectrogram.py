#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
import numpy as np
import matplotlib.pyplot as plt
#import pylab
from scipy.io import wavfile
from scipy.fftpack import fft

filename = "./wav/1523787174095.wav"

samplingFrequency, sound = wavfile.read(filename)

soundDataType = sound.dtype

sound = sound / (2. ** 15)

soundShape = sound.shape
samplePoints = float(sound.shape[0])

signalDuration = sound.shape[0] / sampleFrequency

soundOneChannel = sound[:,0]
"""

""" Does not work
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

filename = "./wav/test_mono_44100Hz_16bit_PCM.wav"
sample_rate, samples = wavfile.read(filename)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
#import cv2
import time

def short_time_fourier_transform(sig, frameSize, overlapFac=0.5, window=np.hanning):
	win = window(frameSize)
	hopSize = int(frameSize - np.floor(overlapFac * frameSize))
	
	# zeros at beginning (thus center of 1st window should be for example nr. 0)
	samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)
	# cols for windowing
	cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
	# zeros at end (thus samples can be fully covered by frames)
	samples = np.append(samples, np.zeros(frameSize))

	frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
	frames *= win

	return np.fft.rfft(frames)

""" scale frequency axis logarithmatically """
def logscale_spec(spec, sr=44100, factor=20.):
	timebins, freqbins = np.shape(spec)

	scale = np.linspace(0, 1, freqbins) ** factor
	scale *= (freqbins - 1) / max(scale)
	scale = np.unique(np.round(scale))

	# create spectrogram with new freq bins
	newspec = np.complex128(np.zeros([timebins, len (scale)]))
	for i in range(len(scale)):
		if i == len(scale) - 1:
			newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
		else:
			newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

	# list center freq of bins
	allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1./sr)[:freqbins+1])
	freqs = []
	for i in range(len(scale)):
		if i == len(scale) - 1:
			freqs += [np.mean(allfreqs[int(scale[i]):])]
		else:
			freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

	return newspec, freqs

""" plot spectrogram """
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
	samplerate, samples = wav.read(audiopath)
	s = short_time_fourier_transform(samples, binsize)

	sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
	ims = 20.*np.log10(np.abs(sshow) / 1e-05)   # amploitude to decimal, 10e-06 == 1e-05

	timebins, freqbins = np.shape(ims)

	plt.figure(figsize=(8, 8))	# plt.figure(figsize=(15, 7.5))
	plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
	# plt.colorbar()

	"""
	plt.xlabel("time (s)")
	plt.ylabel("frequency (hz)")
	plt.xlim([0, timebins-1])
	plt.ylim([0, freqbins])

	xlocs = np.float32(np.linspace(0, timebins-1, 5))
	plt.xticks(xlocs, ["%.02f" % i for i in ((xlocs * len(samples) / timebins) + (0.5 * binsize)) / samplerate])
	ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
	plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
	"""

	plt.axis('off')

	# remove axis labels
	plt.xticks([])
	plt.yticks([])

	if plotpath:
		plt.savefig(plotpath, bbox_inches="tight")
	else:
		plt.show()
		# fig.axes.get_xaxis().set_visible(False)
		# fig.axes.get_yaxis().set_visible(False)

	plt.clf()


if __name__ == '__main__':
	#filename = "./wav/1523787174095.wav"
	#filename = "./wav/test_mono_44100Hz_16bit_PCM.wav"
	filename = "./wav/1528205058755.wav"
	plotstft(filename, plotpath='./spectrogram/{}.png'.format(int(time.time())))