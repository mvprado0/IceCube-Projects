"""

Loop through all the HDF5 files (NuMu and NuE) and organize events into two dictionaries: 
1. Containing the array names 
2. Linking array names to labels (0 if cascade 1 if track)

The event sample is broken up into a training, validation, and test set.

Author: Maria Prado Rodriguez (mvprado@icecube.wisc.edu)

"""
import tensorflow as tf
from tensorflow import keras 

# Helper libraries
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import random
import glob
import argparse

#IceTray
from I3Tray import *
from icecube import icetray, dataio, dataclasses

def hdf5_get(hf, list_q):

    arr_list = []

    for quantity in list_q:
        d = np.array(hf.get(quantity))
        arr_list.append(d)

    return arr_list[0], arr_list[1], arr_list[2], arr_list[3], arr_list[4], arr_list[5], arr_list[6] 

def concatenate_nue_numu(start_e, end_e, elist, start_mu, end_mu, mulist):

    st = [start_e, start_mu]
    nd = [end_e, end_mu]
    files = [elist, mulist]

    for i, lst in enumerate(files):
        if (st[i]==None):
            if (i==0):
                e = lst[:nd[i]]
            else:
                m = lst[:nd[i]]
        elif (nd[i]==None):
            if (i==0):
                e = lst[st[i]:]
            else:
                m = lst[st[i]:]
        else:
            if (i==0):
                e = lst[st[i]:nd[i]]
            else:
                m = lst[st[i]:nd[i]]

    total = e + m

    norm_e = len(e)
    norm_m = len(m)
    norm_prev_e = len(elist)
    norm_prev_m = len(mulist)

    return total, norm_e, norm_m, norm_prev_e, norm_prev_m

def file_loop(fselected, norm_e, norm_m, norm_prev_e, norm_prev_m, name_list, label_dict, energy_dict, weight_dict, track_length, llh, prev_labels, prev_weights):

    counter = 0

    for i, name in enumerate(fselected):

        counter += 1
        
        # Passing the correct normalization factors for NuE and NuMu
        if (i < norm_e):
            norm = norm_e
            norm_prev = norm_prev_e
        else:
            norm = norm_m
            norm_prev = norm_prev_m

        print("Importing file number: " + str(counter))

        name_list, label_dict, energy_dict, weight_dict, track_length, llh, prev_labels, prev_weights = create_dicts( name, name_list, label_dict, energy_dict, weight_dict, track_length, llh, prev_labels, prev_weights, norm, norm_prev)

    return name_list, label_dict, energy_dict, weight_dict, track_length, llh, prev_labels, prev_weights

def create_dicts(name, name_list, label_dict, energy_dict, weight_dict, track_length, llh, prev_labels, prev_weights, norm, norm_prev):
    
    hf = h5py.File( name, 'r')
    quantities = ['labels', 'energies', 'track_length', 'llh', 'prev_labels', 'weights', 'prev_weights']
    labels, energy, track_length0, llh0, prev_labels0, weights, prev_weights0 = hdf5_get(hf, quantities)

    # Matching the order of neutrino event names to the corresponding labels, energies and normalized weights.
    for i, arr in enumerate(hf):
        if 'array_frame' in str(arr): 
            n1 = hf.get(str(arr))
            label_dict.update( { str(arr) : labels[i] } ) 
            energy_dict.update( { str(arr) : energy[i] } )
            weight_dict.update( { str(arr) : (weights[i]/float(norm)) } )
            name_list.append(str(arr))
                            
    hf.close()

    prev_weights0 = prev_weights0/float(norm_prev)
    
    track_length = np.concatenate((track_length, track_length0))
    llh = np.concatenate((llh, llh0))
    prev_labels = np.concatenate((prev_labels, prev_labels0))
    prev_weights = np.concatenate((prev_weights, prev_weights0))

    return name_list, label_dict, energy_dict, weight_dict, track_length, llh, prev_labels, prev_weights

# Main

parser = argparse.ArgumentParser(description='Organize HDF5 files into Numpy arrays.')
parser.add_argument('-e', dest='nue_inputfiles', nargs='+')
parser.add_argument('-m', dest='numu_inputfiles', nargs='+')

args = parser.parse_args()

nue_list = sorted(args.nue_inputfiles)
numu_list = sorted(args.numu_inputfiles)

# All numerical values here are for starting and ending file numbers for training, 
# validation, and testing for electron neutrinos and muon neutrinos.
tr_st_e = 48
tr_d_e = None
tr_st_m = 60 
tr_d_m = 1050
v_st_e = 0
v_d_e = 24
v_st_m = 0
v_d_m = 30
ts_st_e = 24
ts_d_e = 48
ts_st_m = 30
ts_d_m = 60

# Dictionaries, lists, and arrays to fill.
train_label_dict = {}
val_label_dict = {}
test_label_dict = {}
train_energy_dict = {}
val_energy_dict = {}
test_energy_dict = {}
train_weight_dict = {}
val_weight_dict = {}
test_weight_dict = {}
train_name_list = []
val_name_list = []
test_name_list = []
track_length = np.ndarray((0,)) 
llh = np.ndarray((0,)) 
prev_labels = np.ndarray((0,)) 
prev_weights = np.ndarray((0,)) 

print("Training files")
train_list, norm_e, norm_m, norm_prev_e, norm_prev_m = concatenate_nue_numu(tr_st_e, tr_d_e, nue_list, tr_st_m, tr_d_m, numu_list)
train_name_list, train_label_dict, train_energy_dict, train_weight_dict, track_length, llh, prev_labels, prev_weights = file_loop(train_list, norm_e, norm_m, norm_prev_e, norm_prev_m, train_name_list, train_label_dict, train_energy_dict, train_weight_dict, track_length, llh, prev_labels, prev_weights)

print("Validation files")
val_list, norm_e, norm_m, norm_prev_e, norm_prev_m = concatenate_nue_numu(v_st_e, v_d_e, nue_list, v_st_m, v_d_m, numu_list)
val_name_list, val_label_dict, val_energy_dict, val_weight_dict, track_length, llh, prev_labels, prev_weights = file_loop(val_list, norm_e, norm_m, norm_prev_e, norm_prev_m, val_name_list, val_label_dict, val_energy_dict, val_weight_dict, track_length, llh, prev_labels, prev_weights)

print("Test files")
test_list, norm_e, norm_m, norm_prev_e, norm_prev_m = concatenate_nue_numu(ts_st_e, ts_d_e, nue_list, ts_st_m, ts_d_m, numu_list)
test_name_list, test_label_dict, test_energy_dict, test_weight_dict, track_length, llh, prev_labels, prev_weights = file_loop(test_list, norm_e, norm_m, norm_prev_e, norm_prev_m, test_name_list, test_label_dict, test_energy_dict, test_weight_dict, track_length, llh, prev_labels, prev_weights)

np.random.shuffle(train_name_list)
np.random.shuffle(val_name_list)
np.random.shuffle(test_name_list)

name_dict = { 'train' : train_name_list, 'validation' : val_name_list, 'test' : test_name_list }

np.save("name_dict_cut_inicepulses_oscnext.npy", name_dict)
np.save("train_label_dict_cut_inicepulses_oscnext.npy", train_label_dict)
np.save("val_label_dict_cut_inicepulses_oscnext.npy", val_label_dict)
np.save("test_label_dict_cut_inicepulses_oscnext.npy", test_label_dict)
np.save("train_energy_dict_cut_inicepulses_oscnext.npy", train_energy_dict)
np.save("val_energy_dict_cut_inicepulses_oscnext.npy", val_energy_dict)
np.save("test_energy_dict_cut_inicepulses_oscnext.npy", test_energy_dict)
np.save("train_weight_dict_cut_inicepulses_oscnext.npy", train_weight_dict)
np.save("val_weight_dict_cut_inicepulses_oscnext.npy", val_weight_dict)
np.save("test_weight_dict_cut_inicepulses_oscnext.npy", test_weight_dict)
np.save("track_length_oscnext.npy", track_length)
np.save("llh_oscnext.npy", llh)
np.save("prev_labels_oscnext.npy", prev_labels)
np.save("prev_weights_oscnext.npy", prev_weights)

print("Number of events/images: "+ str(len(name_dict['train']) + len(name_dict['validation']) + len(name_dict['test'])))
