# Importing python Library
import pickle, copy, warnings, os, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
warnings.filterwarnings('ignore')
from skfeature.utility.construct_W import construct_W
from scipy.sparse import diags

def fisher_score(X, y):
	"""
	This function implements the fisher score feature selection, steps are as follows:
	1. Construct the affinity matrix W in fisher score way
	2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
	3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
	4. Fisher score for the r-th feature is score = (fr_hat'*D*fr_hat)/(fr_hat'*L*fr_hat)-1
	Input
	-----
	X: {numpy array}, shape (n_samples, n_features)
		input data
	y: {numpy array}, shape (n_samples,)
		input class labels
	Output
	------
	score: {numpy array}, shape (n_features,)
		fisher score for each feature
	Reference
	---------
	He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
	Duda, Richard et al. "Pattern classification." John Wiley & Sons, 2012.
	"""
	# Construct weight matrix W in a fisherScore way
	kwargs = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
	W = construct_W(X, **kwargs)

	# build the diagonal D matrix from affinity matrix W
	D = np.array(W.sum(axis=1))
	L = W
	tmp = np.dot(np.transpose(D), X)
	D = diags(np.transpose(D), [0])
	Xt = np.transpose(X)
	t1 = np.transpose(np.dot(Xt, D.todense()))
	t2 = np.transpose(np.dot(Xt, L.todense()))
	# compute the numerator of Lr
	D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
	# compute the denominator of Lr
	L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
	# avoid the denominator of Lr to be 0
	D_prime[D_prime < 1e-12] = 10000
	lap_score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]

	# compute fisher score from laplacian score, where fisher_score = 1/lap_score - 1
	score = 1.0/lap_score - 1
	return np.transpose(score)


def get_fisher_score(subband, labels):
	# get the subband data
	X_theta, X_alpha = np.array(subband['theta']), np.array(subband['alpha'])
	X_beta, X_gamma = np.array(subband['beta']), np.array(subband['gamma'])
	y = np.array(labels)

	# apply Min-Max scalling on the given data
	scaler = MinMaxScaler()
	scaler.fit(X_theta)
	X_theta = scaler.transform(X_theta)
	scaler.fit(X_alpha)
	X_alpha = scaler.transform(X_alpha)
	scaler.fit(X_beta)
	X_beta = scaler.transform(X_beta)
	scaler.fit(X_gamma)
	X_gamma = scaler.transform(X_gamma)

	# get the fscore for the subband
	fscore_theta, fscore_alpha = fisher_score(X_theta, y), fisher_score(X_alpha, y)
	fscore_beta, fscore_gamma = fisher_score(X_beta, y), fisher_score(X_gamma, y)

	# Total Avearge F-Score (Theta, Alpha, Beta, Gamma)
	final_f_score = (fscore_theta + fscore_alpha + fscore_beta + fscore_gamma)/4
	fvalues = pd.Series(final_f_score)
	fvalues.index = eeg_channels
	fvalues.sort_values(ascending = False)
	fvalues.to_csv('fscore_final.csv')
	# for visualization run the below code
	# fvalues.sort_values(ascending = False).plot.bar(figsize=(10,8))
	df = fvalues.sort_values(ascending = False)
	da = pd.DataFrame(df)
	da.to_csv('channel_rank.csv')
	cr = pd.read_csv('channel_rank.csv')
	sort_channel_name = list(cr['Unnamed: 0'])
	os.remove('channel_rank.csv') # delete the csv file
	return sort_channel_name


#Loading the dataset
def svmclassifier(sorted_channels, data, labels, args):
	# get the features corresponding to the selected channels
	features = []
	for ich in sorted_channels:
		features.append(ich + "_theta")
		features.append(ich + "_alpha")
		features.append(ich + "_beta")
		features.append(ich + "_gamma")
	# get the corresponding data and lables
	x, y = data[features], np.array(labels)

	# Implementing cross validation
	kf = KFold(n_splits = args.no_of_folds, shuffle = False)
	acc_score = []
	for train_index , test_index in kf.split(x):
		x_train, x_test = x.iloc[train_index,:],x.iloc[test_index,:]
		y_train, y_test = y[train_index] , y[test_index]
		model = svm.SVC(kernel = 'poly')
		model.fit(x_train, y_train)
		pred_values = model.predict(x_test)
		#pred_values = model.predict(x_test)
		acc = accuracy_score(pred_values, y_test)
		acc_score.append(acc)
	avg_acc_score = sum(acc_score)/args.no_of_folds
	return avg_acc_score

def growing_phase(sortes_channels, data, labels, args):
	channel = sortes_channels[0]
	# always pass channels as a list
	acc = svmclassifier([channel], data.copy(), labels, args)
	cn_list, sort_cn = [], []
	cn_list.append(channel)
	for i in range(1, len(sortes_channels)):
		cur_ch = sortes_channels[i] # current channel
		cn_list.append(cur_ch) # add into visited channels list
		# get the accuracy of current channels
		cur_acc = svmclassifier(cn_list, data.copy(), labels, args)
		if(cur_acc<acc):
			# remove the current channel if the accuracy is lesser
			cn_list.remove(cur_ch)
		else:
			acc = cur_acc
	print('Accuracy in Growing Phase: ', acc)
	print('No of selected channels in Growing Phase: ', len(cn_list))
	print('Channels selected in Growing Phase: ', cn_list)
	return cn_list

def get_band_data(data):
	theta_feat, alpha_feat, beta_feat, gamma_feat = [], [], [], []
	for ich in channels:
		theta_feat.append(ich + '_theta')
		alpha_feat.append(ich + '_alpha')
		beta_feat.append(ich + '_beta')
		gamma_feat.append(ich + '_gamma')
	theta, alpha = data[theta_feat], data[alpha_feat]
	beta, gamma  = data[beta_feat], data[gamma_feat]
	return theta, alpha, beta, gamma


def main(args):
	# store the subject wise paths for each classification types
	sub_val_paths, sub_ar_paths, sub_four_paths = [], [], []
	for isub in subjects:
		sub_val_paths.append(args.datafiles_path + isub + "/" + isub + '_valence.csv')
		sub_ar_paths.append(args.datafiles_path + isub + "/" + isub + '_arousal.csv')
		sub_four_paths.append(args.datafiles_path + isub + "/" + isub + '_four_class.csv')
	
	# merge all subject wise data and labels
	data_val = pd.concat(map(pd.read_csv, sub_val_paths), ignore_index = True)
	# valence label
	valence_labels = data_val['valence']
	data = data_val.drop('valence', axis=1)
	# get the subbands data
	theta, alpha, beta, gamma = get_band_data(data)
	subband = dict(theta = theta, alpha = alpha, beta = beta, gamma = gamma)
	print("Classification type: valence")
	# get the fisher score for valence class
	sorted_val_channels = get_fisher_score(subband, valence_labels)
	# get the optimal set of channels for valence class
	optimal_val_channels = growing_phase(sorted_val_channels, data.copy(), valence_labels.copy(), args)

	# arousal
	data_ar = pd.concat(map(pd.read_csv, sub_ar_paths), ignore_index = True)
	arousal_labels = data_ar['arousal']
	print("\nClassification type: arousal")
	# get the fisher score for arousal class
	sorted_ar_channels = get_fisher_score(subband, arousal_labels)
	# get the optimal set of channels for valence class
	optimal_ar_channels = growing_phase(sorted_ar_channels, data.copy(), arousal_labels.copy(), args)

	# four class
	data_val = pd.concat(map(pd.read_csv, sub_four_paths), ignore_index = True)
	four_labels = data_val['all']
	print("\nClassification type: four class")
	# get the fisher score for four class
	sorted_four_channels = get_fisher_score(subband, four_labels)
	# get the optimal set of channels for valence class
	optimal_val_channels = growing_phase(sorted_four_channels, data.copy(), four_labels.copy(), args)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# Global Variables
	channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
	            'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
	            'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

	subjects = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11',
				's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21', 's22',
				's23', 's24', 's25', 's26', 's27', 's28', 's29', 's30', 's31', 's32']
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
	parser.add_argument("--no_of_folds", type=int, default=32, help = "No of folds for cross validation.")
	parser.add_argument("--deap_dataset_path", type=str, default=deap_dataset_path, help="DEAP dataset path")
	parser.add_argument("--datafiles_path", type=str, default=datafiles_path, help="Location of subject wise datafiles")
	args = parser.parse_args()
	main(args)