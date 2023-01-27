# DWT based features extraction from ALL Channels

# Importing python Library
import pickle, math, warnings, copy
import utils, mne, argparse, shutil, os
import numpy as np
import pandas as pd
from tqdm import trange
from pywt import wavedec # library for wavelet decomposition
from itertools import chain
from hurst import compute_Hc
from mne.preprocessing import ICA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from mne.filter import filter_data as bandpass_filter
warnings.filterwarnings('ignore')

def SignalPreProcess(eeg_rawdata):
	"""
	:param eeg_rawdata: numpy array with the shape of (n_channels, n_samples)
	:return: filtered EEG raw data
	"""

	assert eeg_rawdata.shape[0] == 32
	eeg_rawdata = np.array(eeg_rawdata)

	info = mne.create_info(ch_names = channels, ch_types = ['eeg' for _ in range(32)], sfreq = 128, verbose=False)
	raw_data = mne.io.RawArray(eeg_rawdata, info, verbose = False)
	raw_data.load_data(verbose = False).filter(l_freq = 0.01, h_freq = 60, method = 'fir', verbose = False)
	#raw_data.plot()

	ica = ICA(n_components = None, random_state = 97, verbose = False)
	ica.fit(raw_data)
	eog_indices, eog_scores = ica.find_bads_eog(raw_data.copy(), ch_name = 'Fp1', verbose = None)
	#ica.plot_scores(eog_scores)
	a = abs(eog_scores).tolist()
	droping_components = 'one'
	if(droping_components == 'one'):
		ica.exclude = [a.index(max(a))]

	else: # find two maximum scores
		a_2 = a.copy()
		a.sort(reverse = True)
		exclude_index = []
		for i in range(0, 2):
			for j in range(0, len(a_2)):
				if(a[i]==a_2[j]):
					exclude_index.append(j)
		ica.exclude = exclude_index
	ica.apply(raw_data, verbose = False)
	# common average reference
	raw_data.set_eeg_reference('average', ch_type = 'eeg')#, projection = True)
	filted_eeg_rawdata = np.array(raw_data.get_data())
	return filted_eeg_rawdata


def signal_pro(input_data):
	print('Data preprocessing (Trial wise):')
	for i in trange(input_data.shape[0]):
		input_data[i] = SignalPreProcess(input_data[i].copy())
	return input_data


def get_features_name(total_features):
	features = []
	for feat in range(0, total_features):
		features.append('feat_' + str(feat+1))
	return features



def emotion_label(labels, class_label):
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


def dwt_features(data):
	'''
	--------------------------------------------------------------------------------------------
	Extracted Discreate Wavelet domain features:
	--------------------------------------------------------------------------------------------
	Mean, Variance, Mode, Median, Skew, Standard deviation, Kurtosis,
	Energy, Average Power, RMS, Shannon Entropy, Approximate Entropy
	Permutation Entropy, Weighted Permutation Entropy, Hurst Exponent,
	Higuchi Fractal Dimension, Petrosian Fractal Dimension, Spectral
	Entropy, Mean of Peak Frequency, Auto Regressive and Auto Regressive
	moving Average model parameters computed on decomposition coefficients
	'''
	coeffs = wavedec(data, 'db1', level = 4)
	delta, theta, alpha, beta, gamma = coeffs

	# Statistical feature, computed from the DWT-based decomposed subbands
	theta_stat = utils.statistical_features(theta, advanced = True)
	alpha_stat = utils.statistical_features(alpha, advanced = True) 
	beta_stat  = utils.statistical_features(beta, advanced = True)
	gamma_stat = utils.statistical_features(gamma, advanced = True)

	# Energy calculation for each band
	theta_energy, alpha_energy  = sum(abs(theta)**2), sum(abs(alpha)**2)
	beta_energy, gamma_energy = sum(abs(beta)**2), sum(abs(gamma)**2)

	# Average power and RMS for each band
	theta_avg_power, theta_rms = utils.avg_and_rms_power(theta)
	alpha_avg_power, alpha_rms = utils.avg_and_rms_power(alpha)
	beta_avg_power, beta_rms = utils.avg_and_rms_power(beta)
	gamma_avg_power, gamma_rms = utils.avg_and_rms_power(gamma)

	# Shannon entropy (shEn)
	theta_ShEn, alpha_ShEn = utils.shannon_entropy(theta), utils.shannon_entropy(alpha)
	beta_ShEn, gamma_ShEn = utils.shannon_entropy(beta), utils.shannon_entropy(gamma)

	# Approximate entropy
	theta_aentropy, alpha_aentropy = utils.app_entropy(theta), utils.app_entropy(alpha)
	beta_aentropy, gamma_aentropy = utils.app_entropy(beta), utils.app_entropy(gamma)

	# Permutation entropy
	theta_pentropy = utils.perm_entropy(theta, normalize = True)
	alpha_pentropy = utils.perm_entropy(alpha, normalize = True)
	beta_pentropy  = utils.perm_entropy(beta, normalize = True)
	gamma_pentropy = utils.perm_entropy(gamma, normalize = True)


	# Weigheted Permutation Entropy
	theta_wpe = utils.weighted_permutation_entropy(theta, order = 3, normalize = False)
	alpha_wpe = utils.weighted_permutation_entropy(alpha, order = 3, normalize = False)
	beta_wpe  = utils.weighted_permutation_entropy(beta, order = 3, normalize = False)
	gamma_wpe = utils.weighted_permutation_entropy(gamma, order = 3, normalize = False)

	# Hurst Exponent(HE): Here we have two paramaters of HE i.e. H and c
	H_theta, c_theta, data_HC_theta = compute_Hc(theta, kind = 'change', simplified = True)
	H_alpha, c_alpha, data_HC_alpha = compute_Hc(alpha, kind = 'change', simplified = True)
	H_beta,  c_beta,  data_HC_beta  = compute_Hc(beta,  kind = 'change', simplified = True)
	H_gamma, c_gamma, data_HC_gamma = compute_Hc(gamma, kind = 'change', simplified = True)

	# Higuchi Fractal dimention
	higuchi_theta = utils.higuchi_fd(theta) # Higuchi fractal dimension for theta
	higuchi_alpha = utils.higuchi_fd(alpha) # Higuchi fractal dimension for alpha
	higuchi_beta  = utils.higuchi_fd(beta)  # Higuchi fractal dimension for beta
	higuchi_gamma = utils.higuchi_fd(gamma) # Higuchi fractal dimension for gamma

	# Petrosian fractal dimension
	petrosian_theta = utils.petrosian_fd(theta) # Petrosian fractal dimension for theta
	petrosian_alpha = utils.petrosian_fd(alpha) # Petrosian fractal dimension for alpha
	petrosian_beta  = utils.petrosian_fd(beta)  # Petrosian fractal dimension for beta
	petrosian_gamma = utils.petrosian_fd(gamma) # Petrosian fractal dimension for gamma
	
	# Auto regressive (AR)
	res_theta = AutoReg(theta,lags = 128).fit()
	res_alpha = AutoReg(alpha,lags = 128).fit()
	res_beta  = AutoReg(beta,lags = 128).fit()
	res_gamma = AutoReg(gamma,lags = 128).fit()
	aic_theta_ar, hqic_theta_ar, bic_theta_ar, llf_theta_ar = res_theta.aic, res_theta.hqic, res_theta.bic, res_theta.llf
	aic_alpha_ar, hqic_alpha_ar, bic_alpha_ar, llf_alpha_ar = res_alpha.aic, res_alpha.hqic, res_alpha.bic, res_alpha.llf
	aic_beta_ar,  hqic_beta_ar,  bic_beta_ar,  llf_beta_ar  = res_beta.aic,  res_beta.hqic,  res_beta.bic,  res_beta.llf
	aic_gamma_ar, hqic_gamma_ar, bic_gamma_ar, llf_gamma_ar = res_gamma.aic, res_gamma.hqic, res_gamma.bic, res_gamma.llf

	# Autoregressive moving Average (ARMA)
	try: arma_theta = ARIMA(theta, order=(5,1,0)).fit()
	except: arma_theta = ARIMA(theta, order=(3, 1,0)).fit()
	try: arma_alpha = ARIMA(alpha, order=(5,1,0)).fit()
	except: arma_alpha = ARIMA(alpha, order=(3,1,0)).fit()
	try: arma_beta = ARIMA(beta, order=(5,1,0)).fit()
	except: arma_beta = ARIMA(beta, order=(3,1,0)).fit()
	try: arma_gamma = ARIMA(gamma, order=(5,1,0)).fit()
	except: arma_gamma = ARIMA(gamma, order=(3,1,0)).fit()
	aic_theta_arma, hqic_theta_arma = arma_theta.aic, arma_theta.hqic
	bic_theta_arma, llf_theta_arma  = arma_theta.bic, arma_theta.llf
	aic_alpha_arma, hqic_alpha_arma = arma_alpha.aic, arma_alpha.hqic
	bic_alpha_arma, llf_alpha_arma = arma_alpha.bic, arma_alpha.llf
	aic_beta_arma,  hqic_beta_arma = arma_beta.aic, arma_beta.hqic
	bic_beta_arma,  llf_beta_arma  = arma_beta.bic, arma_beta.llf
	aic_gamma_arma, hqic_gamma_arma = arma_gamma.aic, arma_gamma.hqic
	bic_gamma_arma, llf_gamma_arma = arma_gamma.bic, arma_gamma.llf

	theta_vector = [theta_energy, theta_avg_power, theta_rms, theta_ShEn, theta_aentropy, theta_pentropy,
					theta_wpe, H_theta, c_theta, higuchi_theta, petrosian_theta, aic_theta_ar, hqic_theta_ar, bic_theta_ar,
					llf_theta_ar, aic_theta_arma, hqic_theta_arma, bic_theta_arma, llf_theta_arma]
	theta_vector = theta_stat + theta_vector

	alpha_vector = [alpha_energy, alpha_avg_power, alpha_rms, alpha_ShEn, alpha_aentropy, alpha_pentropy,
					alpha_wpe, H_alpha, c_alpha, higuchi_alpha, petrosian_alpha, aic_alpha_ar, hqic_alpha_ar, bic_alpha_ar,
					llf_alpha_ar, aic_alpha_arma, hqic_alpha_arma, bic_alpha_arma, llf_alpha_arma]
	alpha_vector = alpha_stat + alpha_vector

	beta_vector = [beta_energy, beta_avg_power, beta_rms, beta_ShEn, beta_aentropy, beta_pentropy,
					beta_wpe, H_beta, c_beta, higuchi_beta, petrosian_beta, aic_beta_ar, hqic_beta_ar, bic_beta_ar,
					llf_beta_ar, aic_beta_arma, hqic_beta_arma, bic_beta_arma, llf_beta_arma]
	beta_vector = beta_stat + beta_vector

	gamma_vector = [gamma_energy, gamma_avg_power, gamma_rms, gamma_ShEn, gamma_aentropy, gamma_pentropy,
					gamma_wpe, H_gamma, c_gamma, higuchi_gamma, petrosian_gamma, aic_gamma_ar, hqic_gamma_ar, bic_gamma_ar,
					llf_gamma_ar, aic_gamma_arma, hqic_gamma_arma, bic_gamma_arma, llf_gamma_arma]
	gamma_vector = gamma_stat + gamma_vector

	feature = [theta_vector, alpha_vector, beta_vector, gamma_vector]
	feature = list(chain.from_iterable(list(feature)))
	return feature


def main(args):
	# load the dataset
	with open(args.deap_dataset_path + args.subject + '.dat', 'rb') as f:
		raw_data = pickle.load(f, encoding = 'latin1')
	# raw_data has two key 'data' and 'labels'
	data = raw_data['data']
	labels = raw_data['labels']
	# we are excluding 3s pre base line i.e. first 3*128 = 384 data points from time series data
	reduced_eeg_data  = data[0:40, 0:32, 384:8064]

	# Signal Preprocessing
	filter_data = signal_pro(reduced_eeg_data.copy())

	# DWT based features
	features = []
	print("DTW based feature extraction:")
	for video in trange(0, 40):
		col = []
		for channel in range(0, 32):
			col.append(dwt_features(filter_data[video, channel].copy()))
		# merge all list in col, 2D list to 1D list
		temp = np.array(list(chain.from_iterable(col)))
		features.append(temp)

	features = np.array(features)

	# make the subject name directory to save the csv file
	subject_path = args.datafiles_path + args.subject
	try:
		os.mkdir(subject_path)
	except:
		# If directory exists then delete that directory
		shutil.rmtree(subject_path)
		# then make the new directory
		os.mkdir(subject_path)

	# create the dataframe
	features_ = get_features_name(features.shape[1])
	df = pd.DataFrame(features, columns = features_)

	# Save the CSV files
	df_valence, df_arousal, df_all = df.copy(), df.copy(), df.copy()
	# add class labels to the last column
	df_valence['valence'] = emotion_label(labels, 'valence')
	df_arousal['arousal'] = emotion_label(labels, 'arousal')
	df_all['all'] = emotion_label(labels, 'all')
	# save the dataframe to csv file
	df_valence.to_csv(subject_path + '/' + args.subject + '_valence.csv', index = False)
	df_arousal.to_csv(subject_path + '/' + args.subject + '_arousal.csv', index = False)
	df_all.to_csv(subject_path + '/' + args.subject + '_all.csv', index = False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Global Variables
	channels = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", 
				"Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
				"PO4", "O2"]

	# 'deap dataset path' put the path in which deap dataset files are kept.
	deap_dataset_path = '/Users/shyammarjit/Desktop/Brain Computer Interface/Deap Dataset/'
	# put the path location of datfiles folder s.t. subject wise folder should contain datafiles
	datafiles_path = '/Users/shyammarjit/Desktop/Brain Computer Interface/HSFCS/code/datafiles/wavelet/'

	parser.add_argument("--subject", type=str, default="s01", help="subject name")
	parser.add_argument("--deap_dataset_path", type=str, default=deap_dataset_path, help="DEAP dataset path")
	parser.add_argument("--datafiles_path", type=str, default=datafiles_path, help="Location of subject wise datafiles")
	args = parser.parse_args()
	main(args)