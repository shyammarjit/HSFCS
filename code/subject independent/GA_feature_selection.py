# Importing python Library
import warnings, pickle, math
import argparse, random, numpy, time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from deap import creator, base, tools, algorithms, GA
from scoop import futures
from sklearn.utils import shuffle
warnings.filterwarnings('ignore')


def getFitness(individual, fold_data, alpha):
	x_train, x_test = fold_data['x_train'], fold_data['x_test']
	y_train, y_test = fold_data['y_train'], fold_data['y_test']
	total_features = int(x_train.shape[1])
	all_features_name = list(x_train.columns)
	if(len(set(individual)) == 1 and list(set(individual))[0] == 0):
		# If all gene values are 0 then return 0
		return 0
	features = []
	for i in range(0, len(individual)):
		if(individual[i]==1):
			features.append(all_features_name[i])
	no_sel_features = len(features)
	_classifier = SVC(kernel = 'poly')
	new_x_train = x_train[features].copy()
	new_x_test = x_test[features].copy()
	_classifier.fit(new_x_train, y_train)
	predictions = _classifier.predict(new_x_test)
	accuracy = accuracy_score(y_true = y_test, y_pred = predictions)
	my_fitness = alpha*accuracy + (1-alpha)*((total_features - no_sel_features)/total_features)
	return (my_fitness,)

def get_final_report(individual, fold_data, clf_type):
	total_features = int(fold_data['x_train'].shape[1])
	all_features_name = list(fold_data['x_train'].columns)
	if(len(set(individual)) == 1 and list(set(individual))[0] == 0):
		# If all gene values are 0 then return 0
		return 0, 0, 0, 0
	features = []
	for i in range(0, len(individual)):
		if(individual[i]==1):
			features.append(all_features_name[i])
	no_sel_features = len(features)
	_classifier = SVC(kernel = 'poly')
	new_x_train = fold_data['x_train'][features].copy()
	new_x_test = fold_data['x_test'][features].copy()
	_classifier.fit(new_x_train, fold_data['y_train'])
	predictions = _classifier.predict(new_x_test)
	y_test = fold_data['y_test']
	if(clf_type=='binary'):
		accuracy = accuracy_score(y_true = y_test, y_pred = predictions)
		prec = precision_score(predictions, y_test)
		recall = recall_score(predictions, y_test)
		f1 = f1_score(predictions, y_test)
		return accuracy, prec, recall, f1
	else:
		accuracy = accuracy_score(y_true = y_test, y_pred = predictions)
		prec = precision_score(predictions, y_test, average = "weighted")
		recall = recall_score(predictions, y_test, average = "weighted")
		f1 = f1_score(predictions, y_test, average = "weighted")
		return accuracy, prec, recall, f1

def kfold(x, y):
	# drop constant features
	x = x.loc[:,x.apply(pd.Series.nunique) != 1]

	# do the scalling
	names = x.columns
	scaler = MinMaxScaler()
	x = scaler.fit_transform(x)
	x = pd.DataFrame(x, columns=names)
	feature_vectors = list(x.columns)
	skf = StratifiedKFold(n_splits=10)
	x = np.array(x)
	y = np.array(y)
	skf.get_n_splits(x, y)
	test_data, train_data, train_label, test_label = [], [], [], []
	for train_index, test_index in skf.split(x, y):
		X_train, X_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
		# convert into dataframe
		X_train = pd.DataFrame.from_records(X_train)
		X_train.columns = feature_vectors
		X_test = pd.DataFrame.from_records(X_test)
		X_test.columns = feature_vectors
		train_data.append(X_train)
		test_data.append(X_test)
		train_label.append(y_train)
		test_label.append(y_test)
	return train_data, test_data, train_label, test_label

# GA

def getHof(population, toolbox, args):
	hof = tools.HallOfFame(args.numPop * args.numGen)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", numpy.mean)
	stats.register("std", numpy.std)
	stats.register("min", numpy.min)
	stats.register("max", numpy.max)

	# Launch genetic algorithm, change the crossover and mutation probability
	pop, log = algorithms.eaSimple(population, toolbox, cxpb = args.cxpb, mutpb = args.mutpb,\
									ngen=args.numGen, stats=stats, halloffame=hof, verbose=False)
	return hof, log # Return the hall of fame

def get_features(optimal_channels, all_features):
	# compute repeated features per channel.
	no_of_feat_per_channel = int(len(all_features)/32)
	features = []
	temp_ch = len(channels[0])
	unique_features = []
	for i in range(0, no_of_feat_per_channel):
		unique_features.append(all_features[i][temp_ch:])
	
	for ich in optimal_channels:
		for feat in unique_features:
			features.append(ich + feat)
	return features


def feature_selection(data, labels, args):
	# 10 fold
	train_data, test_data, train_label, test_label = kfold(data, labels)

	# Gentic Algorithm based toolbox
	creator.create('FitnessMax', base.Fitness, weights = (1.0,))
	creator.create('Individual', list, fitness = creator.FitnessMax)
	toolbox = base.Toolbox() # Create Toolbox
	toolbox.register('attr_bool', random.randint, 0, 1)
	toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_bool, int(train_data[0].shape[1]))
	toolbox.register('population', tools.initRepeat, list, toolbox.individual)
	toolbox.register('mate', tools.cxOnePoint)
	toolbox.register('mutate', tools.mutFlipBit, indpb = 0.1)
	toolbox.register('select', tools.selTournament, tournsize = 7)

	acc_cross, prec_cross, recall_cross, f1_score_cross = [], [], [], []
	print('Accuracy\tPre\tRecall\tF1')
	for i in range(0, 10):
		x_train, x_test, y_train, y_test = train_data[i], test_data[i], train_label[i], test_label[i]
		fold_data = dict(x_train = x_train, x_test = x_test, y_train = y_train, y_test = y_test)
		toolbox.register('evaluate', getFitness, fold_data = fold_data, alpha = args.alpha)
		initial_population = toolbox.population(args.numPop)
		hof, log = getHof(initial_population, toolbox, args)
		best_individual = list(hof)[0]
		if(len(set(y_train))==2):
			clf_type = 'binary'
		else:
			clf_type = 'multi'
		acc, prec, recall, f1_score = get_final_report(best_individual, fold_data, clf_type)
		acc_cross.append(acc)
		prec_cross.append(prec)
		recall_cross.append(recall)
		f1_score_cross.append(f1_score)
		print(float('{:.3f}'.format(acc)), '\t\t', float('{:.3f}'.format(prec)), '\t', float('{:.3f}'.format(recall)),
			'\t', float('{:.3f}'.format(f1_score)))
	acc_cross, prec_cross = np.array(acc_cross), np.array(prec_cross)
	recall_cross, f1_score_cross = np.array(recall_cross), np.array(f1_score_cross)
	acc_mean, prec_mean = np.mean(acc_cross), np.mean(prec_cross)
	recall_mean, f1_mean = np.mean(recall_cross), np.mean(f1_score_cross)
	print('-'*43)
	print(float('{:.3f}'.format(acc_mean)), '\t\t', float('{:.3f}'.format(prec_mean)), '\t', float('{:.3f}'.format(recall_mean)), '\t', float('{:.3f}'.format(f1_mean)))

def get_optimal_channels():
	val = ['AF3', 'Fp2', 'FC2', 'Fz', 'Oz', 'O1', 'CP1', 'FC1']
	ar = ['T7', 'C3', 'C4', 'P4', 'F8', 'PO3', 'FC6', 'P3', 'FC5', 'Fp2', 'PO4']
	four = ['T7', 'C4', 'C3', 'P4', 'Fp1', 'AF3', 'FC6']
	return val, ar, four


def main(args):
	print('='*97)
	print(" "*45, args.subject, " "*45)
	print('='*97,"\n")
	# get the optimal channels name
	optimal_channels = get_optimal_channels()

	# read the CSV files
	# valence
	print("Classification Type: valence\n")
	data_vr = pd.read_csv(datafiles_path + args.subject + '/' + args.subject + '_valence.csv')
	val_labels = data_vr['valence']
	data_vr = data_vr.drop('valence', axis=1)
	data_col = data_vr.columns
	val_features = get_features(optimal_channels[0], data_col)
	val_df = data_vr[val_features]
	feature_selection(val_df, val_labels, args)

	# arousal
	print("\nClassification Type: arousal\n")
	data_ar = pd.read_csv(datafiles_path + args.subject + '/' + args.subject + '_arousal.csv')
	ar_labels = data_ar['arousal']
	data_ar = data_ar.drop('arousal', axis=1)
	ar_features = get_features(optimal_channels[1], data_col)
	ar_df = data_ar[ar_features]
	feature_selection(ar_df, ar_labels, args)

	# four class
	print("\nClassification Type: four class\n")
	data_four = pd.read_csv(datafiles_path + args.subject + '/' + args.subject + '_all.csv')
	four_labels = data_four['all']
	data_four = data_four.drop('all', axis=1)
	four_features = get_features(optimal_channels[2], data_col)
	four_df = data_four[four_features]
	feature_selection(four_df, four_labels, args)

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
	parser.add_argument("--alpha", type =float, default = 0.90, help="weight for classification accuracy")
	parser.add_argument("--numPop", type =int, default = 100, help="size of the population in GA")
	parser.add_argument("--numGen", type =int, default = 50, help="No of generations in GA")
	parser.add_argument("--cxpb", type =float, default = 0.75, help="crossover probability in GA")
	parser.add_argument("--mutpb", type =float, default = 0.30, help="mutation probability in GA")
	parser.add_argument("--tournsize", type =int, default = 7, help="tournsize in GA")
	args = parser.parse_args()
	main(args)
