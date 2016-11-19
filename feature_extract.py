
import pandas as pd
import numpy as np
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import os
from scipy.signal import butter, filtfilt
from os.path import splitext
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks
from obspy.signal import filter
import itertools
from scipy import signal

def windows(x, nperseg, noverlap):
    """
    Calculate windowed FFT, for internal use by scipy.signal._spectral_helper
    This is a helper function that does the main FFT calculation for
    _spectral helper. All input valdiation is performed there, and the data
    axis is assumed to be the last axis of x. It is not designed to be called
    externally. The windows are not averaged over; the result from each window
    is returned.
    Returns
    -------
    result : ndarray
        Array of FFT data
    References
    ----------
    .. [1] Stack Overflow, "Repeat NumPy array without replicating data?",
        http://stackoverflow.com/a/5568169
    Notes
    -----
    Adapted from matplotlib.mlab
    .. versionadded:: 0.16.0
    """
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        step = int(nperseg - noverlap)
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)
    return result


def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata

def corr(data,type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    w,v = np.linalg.eig(C)
    #print(w)
    x = np.sort(w)
    x = np.real(x)
    return x

def calculate_features(file_name):
    f = mat_to_data(file_name)
    fs = f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    time = np.arange(len(eegData))/fs

    # define time windows to look at trends
    t_win = 30 # lets start with 30 second windows
    frac_overlap = .5 # at %50 overlap
    length = np.ceil(t_win*fs)
    overlap = np.ceil(length*frac_overlap)

    # loop over channels
    features_by_channel = []
    for i in range(0,16):
        features = []
        data = eegData[:,i]
        eeg_All = data

        # measurements on all frequencies
        data_squared = np.power(data,2)
        energy_all = np.trapz(data_squared,time)
        var_all = np.var(data)
        f_all, Pxx_all = signal.periodogram(data, fs, detrend='constant', return_onesided=True)

        features.append(energy_all)
        features.append(var_all)
        features.append(Pxx_all)

        # measure psd over time
        data_windowed = windows(eeg_All, length, overlap)
        time_windowed = windows(time, length, overlap)
        time_center = np.median(time_windowed,axis=-1)

        f_win, Pxx_win = signal.periodogram(data_windowed, fs, detrend='constant', return_onesided=True,axis=-1)

        # trends in psd over time
        psd_trend1 = []
        psd_trend2 = []
        for i in range(0,len(f_win)):
            m_Pxx,b_Pxx = np.polyfit(time_center, Pxx_win[:,i], 1)
            Pxx_var = np.var(Pxx_win[:,i])
            psd_trend1.append(m_Pxx)
            psd_trend2.append(Pxx_var)
        features.append(psd_trend1)
        features.append(psd_trend2)

        # subset data by EEG frequency: do any particular type of brain-wave correlate with pre-seizures?
        eeg_Delta = filter.lowpass(data,4, fs, corners=2,zerophase=True)
        eeg_Theta = filter.bandpass(data, 4, 7, fs, corners=2,zerophase=True)
        eeg_Alpha= filter.bandpass(data, 7, 14, fs, corners=2,zerophase=True)
        eeg_Beta = filter.bandpass(data, 15, 30, fs, corners=2,zerophase=True)
        eeg_Gamma = filter.bandpass(data, 30, 100, fs, corners=2,zerophase=True)
        eeg_Mu = filter.bandpass(data, 8, 13, fs, corners=2,zerophase=True)
        eeg_High = filter.bandpass(data, 100, 200, fs, corners=2,zerophase=True)

        data_all_freqs = [eeg_Delta,eeg_Theta,eeg_Alpha,eeg_Beta,eeg_Gamma,eeg_Mu,eeg_High]

        for j in range(0,len(data_all_freqs)):
            data = data_all_freqs[j]
            data_squared = np.power(data,2)

            energy = np.trapz(data_squared,time)
            var = np.var(data)
            data_windowed = windows(data, length, overlap)

            # measure variance
            var = np.var(data_windowed,axis=-1)
            # variance trend
            m_var,b_var = np.polyfit(time_center, var, 1)

            # measure energy
            data_windowed_squared = np.power(data_windowed,2)
            energy_win = np.trapz(data_windowed_squared,axis=-1)

            # energy trend
            m_energy,b_energy = np.polyfit(time_center, energy_win, 1)



path = '/Users/julieschnurr/Desktop/finalprog/data/train_1/'
for fn in os.listdir(path):
        fn1 = fn.split('.')[0]
        label = fn1.split('_')[-1]
        calculate_features(path+fn)
