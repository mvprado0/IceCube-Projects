"""

Generator of desired batch size of events. 
Allows training one batch size at a time rather than having to feed all the events at once. 
When dealing with millions of input events, training without a generator becomes too computationally expensive. 

Author: Maria Prado Rodriguez (mvprado@icecube.wisc.edu)

"""

import numpy as np
import h5py
import tensorflow as tf
from tensorflow import keras

class DNNGenerator(keras.utils.Sequence):

    def __init__(self, steps, hdf5_names, labels, weights, dim=(25, 49, 15), batch_size=128, n_classes=2, n_channels=15):

        self.steps = steps
        self.hdf5_names = hdf5_names
        self.labels = labels
        self.weights = weights
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes

    def __len__(self):
        #Batches per epoch
        return self.steps

    def __getitem__(self, index):

        start = index*self.batch_size
        finish = (index+1)*self.batch_size
        batch_list = self.hdf5_names[start : finish]
        data = np.ndarray((0, 25, 49, self.n_channels))
        data2 = np.ndarray((1, 25, 49, self.n_channels))
        y = np.zeros((len(batch_list)))
        
        for i, name in enumerate(batch_list):

            deeparray = np.zeros((25, 49, self.n_channels))
            hf = h5py.File('/data/user/mvprado/DeepCore_DNN/HDF5/2D_files_cut10_inicepulses/' + name.split("zst_")[1].split("_")[0] + '_coord/dc_arrays_' + name.split("file_")[1] + '.h5', 'r')
          
            arr = hf.get(name)
            arr = np.array(arr)
    
            for s in range(len(arr)):
                ind1 = int(arr[s][0])
                ind2 = int(arr[s][1])
                ind3 = int(arr[s][2])
                val = arr[s][3]
                deeparray[ind1][ind2][ind3] = val
                
            data2[0][:][:][:] = deeparray
            data = np.concatenate((data, data2), axis=0)
            
            y[i] = self.labels.item().get(name)
    
            hf.close()
        y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        return data, y

