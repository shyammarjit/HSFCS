# Importing python Library
import argparse, os
import warnings, copy
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from skfeature.utility.construct_W import construct_W
from scipy.sparse import diags
warnings.filterwarnings('ignore')


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

def get_fisher_score(subBand_data, label):
    theta = np.array(subBand_data['theta'])
    alpha = np.array(subBand_data['alpha'])
    beta = np.array(subBand_data['beta'])
    gamma = np.array(subBand_data['gamma'])
    label = np.array(label)
    
    fscore_theta, fscore_alpha = fisher_score(theta, label), fisher_score(alpha, label)
    fscore_beta, fscore_gamma = fisher_score(beta, label), fisher_score(gamma, label)
    
    # Total Avearge F-Score (Theta, Alpha, Beta, Gamma)
    final_f_score = (fscore_theta + fscore_alpha + fscore_beta + fscore_gamma)/4
    fvalues = pd.Series(final_f_score)
    fvalues.index = channels
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
    os.remove('fscore_final.csv') # delete the csv file
    return sort_channel_name

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
    
#Loading the dataset
def svmclassifier(channel_name, data, labels):
    channel_names = []
    for i in range(0, len(channel_name)):
        draft = channel_name[i]
        channel_names.append(draft + "_alpha")
        channel_names.append(draft + "_beta")
        channel_names.append(draft + "_gamma")
        channel_names.append(draft + "_theta")
    x, y = data[channel_names], np.array(labels)
    # Implementing cross validation
    k = 40
    kf = KFold(n_splits = k, shuffle = False)
    acc_score = []
    for train_index , test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index,:],x.iloc[test_index,:]
        y_train, y_test = y[train_index] , y[test_index]
        model = svm.SVC(kernel = 'poly')
        model.fit(x_train, y_train)
        pred_values = model.predict(x_test)
        #pred_values = model.predict(x_test)
        acc = accuracy_score(pred_values , y_test)
        acc_score.append(acc)
    avg_acc_score = sum(acc_score)/k
    return avg_acc_score

def growing_phase(channel_name, data, labels):
    global global_acc
    cn = channel_name[0]
    acc = svmclassifier([cn], data.copy(), labels)
    cn_list = []
    cn_list.append(cn)
    sort_cn = []
    for i in range(1, len(channel_name)):
        cur_cn = channel_name[i]
        cn_list.append(cur_cn)
        cur_acc = svmclassifier(cn_list, data.copy(), labels)
        if(cur_acc<acc):
            cn_list.remove(cur_cn)
        else:
            acc = cur_acc
    print('Accuracy in Growing Phase: ', acc*100)
    global_acc = acc
    print('No of selected channels in Growing Phase: ', len(cn_list))
    print('Channels selected in Growing Phase: ', cn_list)
    return cn_list

def band_wise_data(data):
    theta_feat, alpha_feat, beta_feat, gamma_feat = [], [], [], []
    for ich in channels:
        theta_feat.append(ich + '_theta')
        alpha_feat.append(ich + '_alpha')
        beta_feat.append(ich + '_beta')
        gamma_feat.append(ich + '_gamma')
    theta = data[theta_feat]
    alpha = data[alpha_feat]
    beta = data[beta_feat]
    gamma = data[gamma_feat]
    subBand_data = dict(theta = theta, alpha = alpha, beta = beta, gamma = gamma)
    return subBand_data

def main(args):
    data_path =  args.datafiles_path + args.subject

    # Read the csv file
    data = pd.read_csv(data_path + '/' + args.subject + '_valence.csv')
    # Get the class label
    labels = data['valence']
    data = data.drop('valence', axis=1)
    # Do Min-Max scalling
    data_col = data.columns
    scaler = MinMaxScaler()
    scaler.fit(np.array(data))
    data_arr = scaler.transform(np.array(data))
    data = pd.DataFrame(data_arr, columns = data_col)

    # Get band wise data
    subBand_data = band_wise_data(data.copy())

    # Valence
    print('Class Label: valence')
    # Sorted channels based on fscore
    sorted_channels = get_fisher_score(subBand_data, labels)

    # Get the optimal channels
    optimal_channels = growing_phase(sorted_channels, data, labels)

    # Arousal
    print('Class Label: arousal')
    data_ar = pd.read_csv(data_path + '/' + args.subject + '_arousal.csv')
    labels = data_ar['arousal'] # Get the class label

    # Sorted channels based on fscore
    sorted_channels = get_fisher_score(subBand_data, labels)

    # Get the optimal channels
    optimal_channels = growing_phase(sorted_channels, data, labels)

    # Four class
    print('Class Label: four class')
    data_four = pd.read_csv(data_path + '/' + args.subject + '_four_class.csv')
    labels = data_four['all'] # Get the class label

    # Sorted channels based on fscore
    sorted_channels = get_fisher_score(subBand_data, labels)

    # Get the optimal channels
    optimal_channels = growing_phase(sorted_channels, data, labels)
        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Global Variables
    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
                'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    # Here 'all' refers for 'four class'
    class_labels = ['valence', 'arousal', 'four_class']

    # put the path location of datfiles folder s.t. subject wise folder should contain datafiles
    datafiles_path = '/Users/shyammarjit/Desktop/Brain Computer Interface/HSFCS/code/datafiles/psd/'

    parser.add_argument("--subject", type=str, default="s01", help="subject name")
    parser.add_argument("--datafiles_path", type=str, default=datafiles_path, help="Location of subject wise datafiles")
    args = parser.parse_args()
    main(args)