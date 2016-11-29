import numpy as np
import itertools
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
import os
import matplotlib.pyplot as plt
# from obspy.signal import filter
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

        features.append(energy_all)
        features.append(var_all)

        # measure psd over time
        data_windowed = windows(eeg_All, length, overlap)
        time_windowed = windows(time, length, overlap)
        time_center = np.median(time_windowed,axis=-1)

        # measure energy
        data_windowed_squared = np.power(data_windowed,2)
        energy_win = np.trapz(data_windowed_squared,axis=-1)

        # energy trend
        m_energy,b_energy = np.polyfit(time_center, energy_win, 1)
        features.append(m_energy)

        f_win, Pxx_win = signal.periodogram(data_windowed, fs, detrend='constant', return_onesided=True,axis=-1)
        f_win = f_win[0:3000]
        Pxx_win = Pxx_win[:,0:3000]
        Pxx_win_win = windows(Pxx_win, 80, 20)
        Pxx_win_win = np.median(Pxx_win_win,axis=-1)
        f_win_win = windows(f_win, 80, 20)
        f_win_win = np.median(f_win_win,axis=-1)

        sk = skew(data_windowed,axis=-1)
        ku = kurtosis(data_windowed,axis=-1)
        var = np.var(data_windowed,axis=-1)

        # trends in psd over time by frequency
        psd_trend1 = []
        psd_trend2 = []

        for i in range(0,len(f_win_win)):
            m_Pxx,b_Pxx = np.polyfit(time_center, Pxx_win_win[:,i], 1)
            Pxx_var = np.var(Pxx_win_win[:,i])
            psd_trend1.append(m_Pxx)
            psd_trend2.append(Pxx_var)
        features.extend(psd_trend1)
        features.extend(psd_trend2)

        # time domain statistical trends over time
        m_sk,b_sk = np.polyfit(time_center, sk, 1)
        m_var,b_var = np.polyfit(time_center, var, 1)
        m_ku,b_ku = np.polyfit(time_center, ku, 1)
        features.append(m_sk)
        features.append(m_var)
        features.append(m_ku)

        # subset data by EEG frequency: do any particular type of brain-wave correlate with pre-seizures?
        # eeg_Delta = filter.lowpass(data,4, fs, corners=2,zerophase=True)
        # eeg_Theta = filter.bandpass(data, 4, 7, fs, corners=2,zerophase=True)
        # eeg_Alpha= filter.bandpass(data, 7, 14, fs, corners=2,zerophase=True)
        # eeg_Beta = filter.bandpass(data, 15, 30, fs, corners=2,zerophase=True)
        # eeg_Gamma = filter.bandpass(data, 30, 100, fs, corners=2,zerophase=True)
        # eeg_Mu = filter.bandpass(data, 8, 13, fs, corners=2,zerophase=True)
        # eeg_High = filter.bandpass(data, 100, 200, fs, corners=2,zerophase=True)
        #
        # data_all_freqs = [eeg_Delta,eeg_Theta,eeg_Alpha,eeg_Beta,eeg_Gamma,eeg_Mu,eeg_High]
        #
        # for j in range(0,len(data_all_freqs)):
        #     print (j)
        #     data = data_all_freqs[j]
        #     data_squared = np.power(data,2)
        #     energy_all = np.trapz(data_squared,time)
        #     var_all = np.var(data)
        #
        #     features.append(energy_all)
        #     features.append(var_all)
        #
        #     # measure psd over time
        #     data_windowed = windows(eeg_All, length, overlap)
        #     time_windowed = windows(time, length, overlap)
        #     time_center = np.median(time_windowed,axis=-1)
        #
        #     # measure energy
        #     data_windowed_squared = np.power(data_windowed,2)
        #     energy_win = np.trapz(data_windowed_squared,axis=-1)
        #
        #     # energy trend
        #     m_energy,b_energy = np.polyfit(time_center, energy_win, 1)
        #     features.append(m_energy)
        #
        #     f_win, Pxx_win = signal.periodogram(data_windowed, fs, detrend='constant', return_onesided=True,axis=-1)
        #     sk = skew(data_windowed,axis=-1)
        #     ku = kurtosis(data_windowed,axis=-1)
        #     var = np.var(data_windowed,axis=-1)
        #
        #     # trends in psd over time by frequency
        #     psd_trend1 = []
        #     psd_trend2 = []
        #     for i in range(0,len(f_win)):
        #         m_Pxx,b_Pxx = np.polyfit(time_center, Pxx_win[:,i], 1)
        #         Pxx_var = np.var(Pxx_win[:,i])
        #         psd_trend1.append(m_Pxx)
        #         psd_trend2.append(Pxx_var)
        #     features.append(psd_trend1)
        #     features.append(psd_trend2)
        #
        #     # time domain statistical trends over time
        #     sk_trend = []
        #     ku_trend = []
        #     var_trend = []
        #     for i in range(0,len(f_win)):
        #         m_sk,b_sk = np.polyfit(time_center, sk, 1)
        #         m_var,b_var = np.polyfit(time_center, var, 1)
        #         m_ku,b_ku = np.polyfit(time_center, ku, 1)
        #         sk_trend.append(m_sk)
        #         var_trend.append(m_var)
        #         ku_trend.append(m_ku)
        #     features.append(sk_trend)
        #     features.append(var_trend)
        #     features.append(ku_trend)
        features_by_channel.append(features)
    means = [np.mean(sub) for sub in zip(*features_by_channel)]
    variances = [np.var(sub) for sub in zip(*features_by_channel)]
    all_features = [val for sublist in features_by_channel for val in sublist]
    means_and_vars = []
    means_only = []
    means_only.extend(means)
    means_and_vars.extend(means)
    all_features.extend(means)
    means_and_vars.extend(variances)
    all_features.extend(variances)
    return np.asarray(all_features), np.asarray(means_and_vars), np.asarray(means_only)

# path = '/Users/julieschnurr/Desktop/finalprog/data/train_1/'
path = '/Users/michaelsommer/Documents/classes/ics635/finalProject/train_1/'
num_files = len(os.listdir(path))
label_array = []
feature_array1 = []
feature_array2 = []
feature_array3 = []


for i in range(0,len(os.listdir(path))):
    fn = os.listdir(path)[i]
    print("file: " + str(i+1) + " out of: " + str(num_files))
    if not fn.startswith('.'):
        fn1 = fn.split('.')[0]
        label = fn1.split('_')[-1]
        try:
            all_features, means_and_vars, means_only = calculate_features(path+fn)
        except ValueError:
            print("corrupted file")
            continue
        feature_array1.append(all_features)
        feature_array2.append(means_and_vars)
        feature_array3.append(means_only)
        label_array.append(label)

feature_array1 = np.array(feature_array1)
feature_array2 = np.array(feature_array2)
label_array = np.array(label_array)

# outfile1 = 'features_train_1_all'
# outfile3 = 'features_train_1_mean_var'
# outfile2 = 'labels_train_1'
# outfile4 = 'features_train_1_mean'

np.savez('labels_train_1', labels = label_array)
np.savez('features_train_1_all', features = feature_array1)
np.savez('features_train_1_mean_var', features = feature_array2)
np.savez('features_train_1_mean', features = feature_array3)

# loadfile1 = 'labels_train_1.npz'
# loadfile2 = 'features_train_1_all.npz'
# loadfile3 = 'features_train_1_mean_var.npz'
# loadfile4 = 'features_train_1_mean.npz'
# data = np.load(loadfile1)
# data1 = np.load(loadfile3)
# data2 = np.load(loadfile2)
# data3 = np.load(loadfile4)
