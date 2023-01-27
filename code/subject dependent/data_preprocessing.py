# Importing python Library
import mne, os, time, pickle, warnings, copy, sys, shutil, argparse
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import signal
from tqdm import trange
from scipy.signal import welch
import matplotlib.pyplot as plt
from scipy.integrate import simps
from mne.preprocessing import ICA
from mne.filter import filter_data as bandpass_filter
warnings.filterwarnings('ignore')

def SignalPreProcess(eeg_rawdata):
    '''
    eeg_rawdata: numpy array with the shape of (n_channels, n_samples)
    return: filtered EEG raw data of shape (n_channels, n_samples)
    '''
    assert eeg_rawdata.shape[0] == 32
    eeg_rawdata = np.array(eeg_rawdata)
  
    info = mne.create_info(ch_names = channels, ch_types = ['eeg' for _ in range(32)], sfreq = 128, verbose=False)
    raw_data = mne.io.RawArray(eeg_rawdata, info, verbose = False) # create MNE raw file
    # Bandpass filter of 4 Hz to 48 Hz
    raw_data.load_data(verbose = False).filter(l_freq = 4, h_freq = 48, method = 'fir', verbose = False)
    # raw_data.plot()
    
    # FAST-ICA with 31 number of components
    ica = ICA(n_components = None, random_state = 97, verbose = False)
    ica.fit(raw_data) # fit the data into ica
    # https://mne.tools/stable/generated/mne.preprocessing.find_eog_events.html?highlight=find_eog_#mne.preprocessing.find_eog_events
    # Take Fp1 channel as the reference channel and find the ICA score to choose artfacts score. 
    eog_indices, eog_scores = ica.find_bads_eog(raw_data.copy(), ch_name = 'Fp1', verbose = None)
    # ica.plot_scores(eog_scores)
    a = abs(eog_scores).tolist()
    droping_components = 'one'
    if(droping_components == 'one'): # find one maximum score
        ica.exclude = [a.index(max(a))] # exclude the maximum index
            
    else: # find two maximum scores
        a_2 = a.copy()
        a.sort(reverse = True)
        exclude_index = []
        for i in range(0, 2):
            for j in range(0, len(a_2)):
                if(a[i]==a_2[j]):
                    exclude_index.append(j)
        ica.exclude = exclude_index # exclude these two maximum indeces
    ica.apply(raw_data, verbose = False) # apply the ICA 
    # common average reference
    raw_data.set_eeg_reference('average', ch_type = 'eeg')#, projection = True)
    filted_eeg_rawdata = np.array(raw_data.get_data()) # fetch the data from MNE.
    return filted_eeg_rawdata

def signal_pro(input_data):
    print("Trail preprocessing:")
    for i in trange(input_data.shape[0]): # for each video sample call SignalPreProcess
        input_data[i] = SignalPreProcess(input_data[i].copy())
    return input_data

def bandpower(input_data, band):
    band = np.asarray(band)
    low, high = band # band is the tuple of (low, high)
    nperseg = (2 / low) * sf
    # Compute the modified periodogram (Welch)
    freqs, psd = welch(input_data, sf, nperseg = nperseg)
    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    return np.mean(psd[idx_band]) # mean of the frequency bands

def emotion_label(labels, class_label):
    '''
    This function gives the valence/arousal and HVHA/HVLA/LAHV/LALV class labels
    '''
    em_labels = []
    if(class_label == 'valence'):
        for i in range(0, labels.shape[0]):
            if (labels[i][0]>5): # high valence
                em_labels.append(1)
            else: # low valence
                em_labels.append(0)
        return em_labels
    elif(class_label == 'arousal'):
        for i in range(0, labels.shape[0]):
            if (labels[i][1]>5): # high arousal
                em_labels.append(1)
            else: # low arousal
                em_labels.append(0)
        return em_labels
    elif(class_label == 'all'):
        for i in range(0, labels.shape[0]):
            if (labels[i][0]>5): # high valence
                if(labels[i][1]>5): # high arousal
                    em_labels.append(1) # HVHA
                else:
                    em_labels.append(0) # HVLA
            else: # low valence
                if(labels[i][1]>5): # high arousal
                    em_labels.append(2) # LVHA
                else: # low arousal
                    em_labels.append(3) # LVLA
        return em_labels
    
def get_csv_file(subject, filter_data, labels, datafiles_path):
    theta, alpha, beta, gamma = [], [], [], []
    theta_feat, alpha_feat, beta_feat, gamma_feat = [], [], [], []
    print("Feature Extraction:")
    for video_no in trange (len(filter_data)):
        for channel_no in range (0, len(filter_data[0])):
            temp = copy.deepcopy(filter_data[video_no, channel_no])
            
            # PSD Features
            theta.append(bandpower(temp, theta_band_range))
            alpha.append(bandpower(temp, alpha_band_range))
            beta.append(bandpower(temp, beta_band_range))
            gamma.append(bandpower(temp, gamma_band_range))
            
    # PSD feature matrix
    theta = np.reshape(theta, (40, 32)) # 40 videos and 32 channels for theta band power
    alpha = np.reshape(alpha, (40, 32))
    beta = np.reshape(beta, (40, 32))
    gamma = np.reshape(gamma, (40, 32))
    
    # Add features name in tne dataframe
    for i in range(0, len(channels)):
        theta_feat.append(channels[i] + '_theta')
        alpha_feat.append(channels[i] + '_alpha')
        gamma_feat.append(channels[i] + '_gamma')
        beta_feat.append(channels[i] + '_beta')
    
    df_theta = pd.DataFrame(theta, columns = theta_feat)
    df_alpha = pd.DataFrame(alpha, columns = alpha_feat)
    df_beta  = pd.DataFrame(beta, columns = beta_feat)
    df_gamma = pd.DataFrame(gamma, columns = gamma_feat)
    
    # make the subject name directory to save the csv file
    subject_path = datafiles_path + subject
    try:
        os.mkdir(subject_path)
    except:
        # If directory exists then delete that directory
        shutil.rmtree(subject_path)
        # then make the new directory
        os.mkdir(subject_path)
    
    # Save the CSV files
    frames = [df_theta, df_alpha, df_beta, df_gamma]
    all_bands = pd.concat(frames, axis = 1) # join these 4 dataframes columns wise, rows are fixed
    all_bands_valence, all_bands_arousal, all_bands_all = all_bands.copy(), all_bands.copy(), all_bands.copy()
    all_bands_valence['valence'] = emotion_label(labels, 'valence')
    all_bands_arousal['arousal'] = emotion_label(labels, 'arousal')
    all_bands_all['four_class'] = emotion_label(labels, 'all')
    all_bands_valence.to_csv(subject_path + '/' + subject + '_valence.csv', index = False, encoding = 'utf-8-sig')
    all_bands_arousal.to_csv(subject_path + '/' + subject + '_arousal.csv', index = False, encoding = 'utf-8-sig')
    all_bands_all.to_csv(subject_path + '/' + subject + '_four_class.csv', index = False, encoding = 'utf-8-sig')

def main(args):
    # load the dataset
    with open(args.deap_dataset_path + args.subject + '.dat', 'rb') as f:
        raw_data = pickle.load(f, encoding = 'latin1')
    # raw_data has two key 'data' and 'labels'
    data = raw_data['data']
    labels = raw_data['labels']
    # we are excluding 3s pre base line i.e. first 3*128 = 384 data points from time series data
    reduced_eeg_data  = data[0:40, 0:32, 384:8064]
    filter_data = signal_pro(reduced_eeg_data.copy())
    get_csv_file(args.subject, filter_data, labels, args.datafiles_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Global Variables
    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
                'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    # SubBands range
    theta_band_range = (4, 8)   # drownsiness, emotional connection, intuition, creativity
    alpha_band_range = (8, 12)  # reflection, relaxation
    beta_band_range = (12, 30)  # concentration, problem solving, memory
    gamma_band_range = (30, 48) # cognition, perception, learning, multi-tasking

    sf = 128 # sampling frequency 128 Hz
    eeg_channels = np.array(channels)
    # Here 'all' refers for 'four class'
    class_labels = ['valence', 'arousal', 'all']

    # 'deap dataset path' put the path in which deap dataset files are kept.
    deap_dataset_path = '/Users/shyammarjit/Desktop/Brain Computer Interface/Deap Dataset/'
    # put the path location of datfiles folder s.t. subject wise folder should contain datafiles
    datafiles_path = '/Users/shyammarjit/Desktop/Brain Computer Interface/HSFCS/code/datafiles/psd/'

    parser.add_argument("--subject", type=str, default="s01", help="subject name")
    parser.add_argument("--deap_dataset_path", type=str, default=deap_dataset_path, help="DEAP dataset path")
    parser.add_argument("--datafiles_path", type=str, default=datafiles_path, help="Location of subject wise datafiles")
    args = parser.parse_args()
    main(args)