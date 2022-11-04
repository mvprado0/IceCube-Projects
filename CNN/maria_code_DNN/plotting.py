"""

Plotting script for Convolutional Neural Network output. 

Author: Maria Prado Rodriguez (mvprado@icecube.wisc.edu)

"""


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import h5py
import argparse

def plot_confusion_matrix(method, values, threshold, labels, weights, job_num, cmap=plt.cm.Purples):
    
    true_t, false_t, true_c, false_c = acc_calculator(values, threshold, labels, weights, w_label=1)
    classes = ['Tracks', 'Cascades']
    
    cm = np.ndarray((2,2))
    cm[0][0] = round(true_t, 2)
    cm[1][0] = round(false_t, 2)
    cm[0][1] = round(false_c, 2)
    cm[1][1] = round(true_c, 2)

    np.set_printoptions(precision=2)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label', labelpad=1)
    fmt = '.2f'
    thresh = cm.max() / 2.
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
            ha="center", va="center",
            color="white" if cm[i, j] > 0.5 else "black")
    
    plt.savefig("confusion_matrix_" + method + "_" + job_num + ".png")

def acc_calculator(predictions, thresh, test_labels, weights=None, w_label=0):
   
    true_track = 0
    false_track = 0
    true_cascade = 0
    false_cascade = 0
    track_num = 0
    casc_num = 0

    # Unweighted events
    if w_label == 0:
        for i, pred in enumerate(predictions):
            if test_labels[i] == 1:

                # track_num and casc_num are the total number of true tracks and true cascades in the test sample, respectively.
                track_num += 1

                if pred >= thresh:
                    true_track += 1
                elif pred < thresh:
                    false_cascade += 1

            elif test_labels[i]==0:

                casc_num += 1

                if pred >= thresh:
                    false_track += 1
                elif pred < thresh:
                    true_cascade += 1

    # Weighted events
    elif w_label == 1:
        for i, pred in enumerate(predictions):
            if test_labels[i] == 1:

                track_num += weights[i]

                if pred >= thresh:
                    true_track += weights[i]
                elif pred < thresh:
                    false_cascade += weights[i]

            elif test_labels[i]==0:

                casc_num += weights[i]

                if pred >= thresh:
                    false_track += weights[i]
                elif pred < thresh:
                    true_cascade += weights[i]
    
    true_t = float(true_track)/track_num
    false_t = float(false_track)/casc_num
    true_c = float(true_cascade)/casc_num
    false_c = float(false_cascade)/track_num
            
    
    return true_t, false_t, true_c, false_c

def calc_roc(thresholds, preds, labels, weights):

    true_pos_t = []
    false_pos_t = []

    for thresh in thresholds:
        true_t, false_t, true_c, false_c = acc_calculator(preds, thresh, labels, weights, w_label=1)
        true_pos_t.append(true_t)
        false_pos_t.append(false_t)
    
    auc = -1 * np.trapz(true_pos_t, false_pos_t)

    return true_pos_t, false_pos_t, auc

def plot_roc(th_nn, th_tr, th_llh, th_bdt, preds, tr, llh, bdt, test_l, prev_l, bdt_l, w, prev_w, bdt_w, job_num):

    # Calculate percent of true positive tracks and false positive tracks
    true_pos_t, false_pos_t, area_nn = calc_roc(th_nn, preds, test_l, w)
    true_pos_t_tr, false_pos_t_tr, area_tr = calc_roc(th_tr, tr, prev_l, prev_w)
    true_pos_t_llh, false_pos_t_llh, area_llh = calc_roc(th_llh, llh, prev_l, prev_w)
    true_pos_t_bdt, false_pos_t_bdt, area_bdt = calc_roc(th_bdt, bdt, bdt_l, bdt_w)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(false_pos_t, true_pos_t,'r-', label='CNN AUC: {:.4f}'.format(area_nn))
    ax.plot(false_pos_t_tr, true_pos_t_tr,'-', color='royalblue', label='Track Length AUC: {:.4f}'.format(area_tr))
    ax.plot(false_pos_t_llh, true_pos_t_llh,'-', color='darkorange', label='LLH AUC: {:.4f}'.format(area_llh))
    ax.plot(false_pos_t_bdt, true_pos_t_bdt,'g-', label='BDT AUC: {:.4f}'.format(area_bdt))
    ax.plot((0,1), (0,1), '--', color='k')
    ax.set_xlabel('False Positive',fontsize=20)
    ax.set_ylabel('True Positive',fontsize=20)
    ax.tick_params(labelsize=17)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('ROC Graph - Tracks', fontsize=20)
    ax.grid()
    ax.legend(loc=4, fontsize=15)

    fig.savefig("ROC_tracks_" + job_num + ".png")

def plot_energies(energies, predictions, threshold, test_labels, weights, job_num):

    # Percentage of tracks vs energy
    x_en = np.logspace(1,2.1,10)
    length = len(x_en)
    mask_list = []
    y_tracks = []
    y_cascades = []

    for i in range(len(x_en)-1):
        mask = (energies > x_en[i]) & (energies <= x_en[i+1])
        mask_list.append(mask)

    for t, mask in enumerate(mask_list):
        true_t, false_t, true_c, false_c = acc_calculator(predictions[mask], threshold, test_labels[mask], weights, w_label=1)
        y_tracks.append(true_t)
        y_cascades.append(false_t)

    fig_en = plt.figure(figsize=(8,8))
    ax2 = fig_en.add_subplot(111)
    ax2.plot(x_en[0:(length-1)], y_tracks, label='Tracks')
    ax2.plot(x_en[0:(length-1)], y_cascades, label='Cascades')
    ax2.set_xlim((10, 100))
    ax2.set_ylim((0, 1))
    ax2.grid()
    ax2.legend(loc=2, fontsize=12)
    ax2.set_xlabel('Neutrino Energy (GeV)', fontsize=15)
    ax2.set_ylabel('Percentage of classified tracks (Threshold: '+ str(threshold) + ')', fontsize=15)
    ax2.tick_params(labelsize=17)
    fig_en.savefig("energies_" + job_num + ".png")

def plot_predictions(predictions, weights, threshold, job_num):
    
    fig_hist = plt.figure(figsize=(7,7))
    ax = fig_hist.add_subplot(111)
    ax.hist(predictions, bins=100, range=(0,1), weights=weights, density=True, histtype='step', color='red')
    ax.set_xlabel('Neural Network Predictions (Track-like)', fontsize=20)
    ax.set_ylabel('Normalized Weighted Counts (%)', fontsize=20)
    ax.tick_params(labelsize=17)
    ax.axvline(x=threshold)
    fig_hist.savefig("NN_predictions_plot_" + job_num + ".png")

# Main

# Parameters
parser = argparse.ArgumentParser(description='Training a CNN.')
parser.add_argument('-j', dest='job_num', type=str) 
parser.add_argument('-t', dest='threshold', type=float) 
args = parser.parse_args()

# Load files needed
name_dict = np.load("name_dict_cut_inicepulses_oscnext.npy", allow_pickle=True)
test_label_dict = np.load("test_label_dict_cut_inicepulses_oscnext.npy", allow_pickle=True)
test_energy_dict = np.load("test_energy_dict_cut_inicepulses_oscnext.npy", allow_pickle=True)
test_weight_dict = np.load("test_weight_dict_cut_inicepulses_oscnext.npy", allow_pickle=True)
track_length = np.load("track_length_oscnext.npy", allow_pickle=True)
llh = np.load("llh_oscnext.npy", allow_pickle=True)
prev_labels = np.load("prev_labels_oscnext.npy", allow_pickle=True)
prev_weights = np.load("prev_weights_oscnext.npy", allow_pickle=True)
predictions = np.load("NN_predictions_2D_InIcePulses0.npy", allow_pickle=True)
test_labels = np.load("NN_labels_2D_InIcePulses0.npy", allow_pickle=True)

# Thresholds chosen by each PID method option
trl_bf = 35
llh_bf = 2
bdt_bf = 0.45

# Creating arrays of classification thresholds to scan over to create an ROC curve measuring the performance of classified track-like events
# Four threshold arrays for the four PID method options: LLH, Track length, BDT, and CNN
t_llh = np.logspace(-4, np.log10(max(llh)),2)  #100)
t_tr = np.logspace(-2, np.log10(max(track_length)),2)  #100)
z = np.zeros(1,)
thresholds_llh = np.concatenate((z,t_llh))
thresholds_tr = np.concatenate((z,t_tr))
thresholds_nn = np.arange(0, 1.01, 0.01)
thresholds_bdt = np.arange(0, 1.01, 0.01)

test_size = len(name_dict.item().get('test'))
print("Test events: ")
print(test_size)

energies = np.ndarray((test_size))
weights = np.ndarray((test_size))

# Fill in energy and weight arrays with corresponding values
for i, name in enumerate(name_dict.item().get('test')):
    energies[i] = test_energy_dict.item().get(name)
    weights[i] = test_weight_dict.item().get(name)

# Apply cut to account for rounding performed in the calculation of the test steps when evaluating the test sample
energies = energies[0:predictions.shape[0]]
weights = weights[0:predictions.shape[0]]

# Output from the XGBoost BDT current DeepCore PID method for comparison
hf = h5py.File('pid_model_test_12FEB20_osc_5vars_no-frac_scores.hdf5', 'r')
bdt_predicted_scores = np.array(hf.get('pred_scores'))
bdt_labels = np.array(hf.get('true_scores'))
bdt_weights = np.array(hf.get('weights'))

# Plot the CNN output predictions on the test sample
plot_predictions(predictions, weights, args.threshold, args.job_num)

# Plot ROC curve for track percentages
plot_roc(thresholds_nn, thresholds_tr, thresholds_llh, thresholds_bdt, predictions, track_length, llh, bdt_predicted_scores, test_labels, prev_labels, bdt_labels, weights, prev_weights, bdt_weights, args.job_num)

# Confusion matrices for the chosen threshold of each PID method option
plot_confusion_matrix('CNN', predictions, args.threshold, test_labels, weights, args.job_num)
plot_confusion_matrix('track_length', track_length, trl_bf, prev_labels, prev_weights, args.job_num)
plot_confusion_matrix('LLH', llh, llh_bf, prev_labels, prev_weights, args.job_num)
plot_confusion_matrix('BDT', bdt_predicted_scores, bdt_bf, bdt_labels, bdt_weights, args.job_num)

# Plot the percentage of tracks and cascades classified as tracks versus the event truth energy
plot_energies(energies, predictions, args.threshold, test_labels, weights, args.job_num)
