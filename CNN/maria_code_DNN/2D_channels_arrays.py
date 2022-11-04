"""

Make 3D (2D plus channels dimension) numpy arrays for IceCube DeepCore detector and store them in an HDF5 file. 
One HDF5 file per I3 file. Will contain one array per physics frame/event.

Author: Maria Prado Rodriguez (mvprado@icecube.wisc.edu)

"""


# Helper libraries
import numpy as np
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import h5py
import ntpath 
import os
import argparse

# IceTray
from I3Tray import *
from icecube import icetray, dataio, dataclasses
from icecube.icetray import I3Frame

def path_def(path):
    head, tail = ntpath.split(path)
    if len(tail) == 0:
        return ntpath.basename(head)
    else:
        return tail

# Fills the DeepCore array
def organize(array, t, strnum, domnum, charge):
    
    # DeepCore and surrounding IceCube strings.
    strings = [79, 80, 81, 82, 83, 84, 85, 86, 26, 27, 35, 36, 37, 45, 46]
    # Positioning of DOMs within the array.
    IC_spacing = [0, 3, 5, 8, 10, 13, 15, 18, 20, 23, 25, 28, 30, 33, 35, 38, 40, 43, 45, 48]
    IC_doms = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
    IC_doms = np.array(IC_doms)
    IC_spacing = np.array(IC_spacing)

    for i, s in enumerate(strings):
        mask = strnum == s
        # Check if string was not hit.
        if len(domnum[mask]) == 0:
            pass
                
        else:    
            # Check if string was hit multiple times. 
            for a, h in enumerate(domnum[mask]):
               
                # Check for IceCube strings and DeepCore dust layer.
                if (s < 79 and (40 < domnum[mask][a] < 61)) or (s >= 79 and (11 < domnum[mask][a] < 61)):
                    if (s >= 79):
                        if array[t][h-12][i] != 0:
                            array[t][h-12][i] = array[t][h-12][i] + charge[mask][a]
                        else:
                            array[t][h-12][i] = charge[mask][a]
                    elif (s < 79):
                        mask2 = IC_doms == domnum[mask][a]
                        z = int(IC_spacing[mask2])
                        if array[t][z][i] != 0:
                            array[t][z][i] = array[t][z][i] + charge[mask][a]
                        else:
                            array[t][z][i] = charge[mask][a]
                
    return array

def insideDC(x, y, z):
    # 50.0 is the x-coordinate of String 36 -- centering circle around string 36
    xr = x - 50.0 
    # -38.0 is the y-coordinate of String 36
    yr = y + 38 

    # Z-direction filters out events that occur at the edge of the IceCube strings and appear smaller because of the Dust layer cutoff.
    return z < 200 and z > -500 and xr**2 + yr**2 < 10000

def make_np_arr(inputs, names):

    for i in range(len(inputs)):

        arr = np.array(inputs[i])
        hf.create_dataset(names[i], data=arr)

# Main

parser = argparse.ArgumentParser(description='Setting options for the type of events desired.')
parser.add_argument('-i', dest='i3file', type=str) # I3 file name
parser.add_argument('-p', dest='pulsetype', type=str) # Pulse type (must be of type mapmask) ex. "InIcePulses", "OfflinePulsesHLC"
parser.add_argument('-a', dest='particletype', type=str) # Type of particle ex. NuMu, NuE
args = parser.parse_args()

labels = []
energies = []
frame_type = []
track_length = []
llh = []
prev_labels = []
weights = []
prev_weights = []
mask_list = []
strnum = []
domnum = []
charge = []
time = []

i3file_name = path_def(args.i3file)

# Opens HDF5 file to write 
file0 = dataio.I3File(args.i3file)
hf = h5py.File('dc_arrays_' + i3file_name + '_' + args.particletype +'_'+ args.pulsetype +'.h5', 'w')
print(os.getcwd())
print("Opened the I3 file.")

for a, frame in enumerate(file0):

    frame_type.append(frame.Stop)

    # Check if Q or P frame (only make arrays for P frames). 
    if frame.Stop == I3Frame.Physics:
        
        if frame_type[a-1] == I3Frame.Physics:
        
            pass

        else:

            hitmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, args.pulsetype)
            tree = frame["I3MCTree"]

            # Store event track length and LLHR of track+cascade fit to cascade only fit. 
            # These were PID methods used in previous analyses. 
            # Store values to compare with CNN performance later.
            length = frame["L7_reconstructed_track_length"].value
            dllh = frame["retro_crs_prefit__zero_dllh"]["median"]
            track_length.append(length)
            llh.append(dllh)

            # Store other important event information.
            particle = tree[0].pdg_encoding
            cc_or_nc = frame['I3MCWeightDict']['InteractionType']
            prev_weights.append(frame["I3MCWeightDict"]["weight"])
            primary = tree.get_primaries()[0]
            vtx = primary.pos

            # Previous weights and labels refer to the oscillated event weights and labels of the full sample before any cuts. 
            # They are stored to match the track length and LLHR events for plotting later. 
            if particle == 12 or particle == -12:
                prev_labels.append(0)
            elif (particle == 14 or particle == -14) and cc_or_nc == 2.0:
                prev_labels.append(0)
            elif (particle == 14 or particle == -14) and cc_or_nc == 1.0:
                prev_labels.append(1)
           
            # Energy cut.
            if round(tree[0].energy) >= 10.0:
                for (om, hitlist) in hitmap:
                    for hit in hitlist:
                        strnum.append(om[0])
                        domnum.append(om[1])
                        charge.append(hit.charge)
                        time.append(hit.time)

                # At least 10 hits and inside sphere of radius 100 m for DeepCore events only.
                if (len(charge) >= 10) and (insideDC(vtx.x, vtx.y, vtx.z)): 
                    energies.append(round(tree[0].energy))
                    # Unoscillated weights for the training only.
                    weights.append(frame["I3MCWeightDict"]["weight_no_osc"])
                    
                    # Track or Cascade: NuE_CC = cascade (0), NuE_NC = cascade (0), NuMu_CC = track (1), NuMu_NC = cascade (0)
                    # Weights, energies, and labels for the events that will be fed to the CNN
                    if particle == 12 or particle == -12:
                        labels.append(0)
                    elif (particle == 14 or particle == -14) and cc_or_nc == 2.0:
                        labels.append(0)
                    elif (particle == 14 or particle == -14) and cc_or_nc == 1.0:
                        labels.append(1)

                    strnum = np.array(strnum)
                    domnum = np.array(domnum)
                    charge = np.array(charge)
                    time = np.array(time)

                    index = np.argsort(time)
                    strnum = strnum[index]
                    domnum = domnum[index]
                    charge = charge[index]
                    time = time[index]
                    
                    # Time windows have a 30 nsec spacing.
                    mask_edges = np.linspace(9900,10600,24)
                    mask_edges = np.append(mask_edges,11000)

                    for i in range(len(mask_edges)-1):
                        mask = (time > mask_edges[i]) & (time <= mask_edges[i+1])
                        mask_list.append(mask)

                    # IceCube DeepCore Array: (t, z, channels).
                    deeparray = np.zeros((25, 49, 15))

                    for t, mask in enumerate(mask_list):
                        # Filling the DeepCore array.
                        deeparray = organize(deeparray, t, strnum[mask], domnum[mask], charge[mask])

                    # Create an array that contains the nonzero coordinates of deeparray.
                    coor_array = np.nonzero(deeparray)
                    arr2 = np.ndarray((0,4))

                    for m in range(len(coor_array[0])):
                        arr = np.ndarray((1,3))
                        for i in range(len(coor_array)):
                            arr[0][i] = coor_array[i][m]
                                    
                        a1 = int(arr[0][0])
                        b1 = int(arr[0][1])
                        c1 = int(arr[0][2])

                        val = np.ndarray((1,1))
                        val[0][0] = deeparray[a1][b1][c1]
                        arr = np.concatenate((arr, val), axis=1)
                        arr2 = np.concatenate((arr2, arr), axis=0)

                    # Stores and names each event array inside the HDF5 file.
                    hf.create_dataset('array_frame' + str(a+1).zfill(6) + '_file_' + i3file_name + '_' + args.particletype + '_' + args.pulsetype, data=arr2)

                else:
                    pass
            else:
                pass
    else:
        pass

inputs = [labels, prev_labels, prev_weights, energies, weights, track_length, llh]
names = ['labels', 'prev_labels', 'prev_weights', 'energies', 'weights', 'track_length', 'llh']

make_np_arr(inputs, names)

hf.close()
