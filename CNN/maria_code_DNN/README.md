
# DeepCore Particle Identification (PID) using a Convolutional Neural Network (CNN)

Train, validate, and test a Convolutional Neural Network for IceCube [DeepCore](https://arxiv.org/abs/1109.6096) Particle Identification (PID). There are two outputs for this classification network: 1) Tracks or 2) Cascades. PID helps prevent the dilution of the signal between event signatures and it is a critical component of DeepCore analyses. 

## Running the scripts


Make 3D (2D plus channels dimension) numpy arrays for each IceCube DeepCore neutrino event.

```
./create_many_HDF5.sh <pdg_code_neutrino_x_10000> <file_number> <neutrino_flavor>
```

Then organize all the events into a dictionary holding the array names (to be called on by the training script later). Another dictionary will hold the corresponding labels (0 if cascade, 1 if track) to each array name in the correct order. 

The event sample is broken up into a training, validation, and test set.

```
./run_data_organizer.sh
```

Train the CNN in batches by calling `generator2D` to produce a batch of 2D-plus-channels HDF5 arrays along with their corresponding truth labels.
For CNN training in the cluster with a GPU run the following:

```
./run_training_GPU_2D.sh
```

For training from the command line with a CPU set python virtual environment and run the following:

```
python 2D_training.py -b <batch_size> -p <epoch_number> -c <conv1_filters> -c2 <conv2_filters> -d <dense1_units> -d2 <dense2_units> -j <run_name>
```

To plot the CNN predictions, the ROC curve, percentage of classified tracks vs energy, and the confusion matrices for specific decision boundaries, run the following:

```
python plotting.py -t <cnn_decision_boundary> -j <run_name>
```
