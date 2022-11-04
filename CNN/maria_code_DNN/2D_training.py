"""

Train a Convolutional Neural Network for classification of different event signatures in IceCube DeepCore. 
The network looks to distinguish track-like signatures from cascade-like signatures. 

Author: Maria Prado Rodriguez (mvprado@icecube.wisc.edu)

"""


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import h5py
import argparse
import tensorflow as tf
from tensorflow import keras 
from generator2D import DNNGenerator
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.backend import set_session

# Main

# Parameters
parser = argparse.ArgumentParser(description='Training a CNN.')
parser.add_argument('-b', dest='batch_size', type=int) 
parser.add_argument('-p', dest='epoch_num', type=int) 
parser.add_argument('-c', dest='conv', type=int) 
parser.add_argument('-c2', dest='conv2', type=int)
parser.add_argument('-d', dest='dense', type=int) 
parser.add_argument('-d2', dest='dense2', type=int) 
parser.add_argument('-j', dest='job_num', type=str) 
args = parser.parse_args()

params = {'dim': (25, 49, 15),
          'batch_size': args.batch_size,
          'n_classes': 2,
          'n_channels': 15}

# For evaluating the test sample at the end
predictions = np.ndarray((0,2))
test_labels = np.ndarray((0,2))

# Load dictionary that links HDF5 file event names to their corresponding label for training, validation, and testing
name_dict = np.load("name_dict_cut_inicepulses_oscnext.npy", allow_pickle=True)
train_label_dict = np.load("train_label_dict_cut_inicepulses_oscnext.npy", allow_pickle=True)
val_label_dict = np.load("val_label_dict_cut_inicepulses_oscnext.npy", allow_pickle=True)
test_label_dict = np.load("test_label_dict_cut_inicepulses_oscnext.npy", allow_pickle=True)

train_size = len(name_dict.item().get('train'))
val_size = len(name_dict.item().get('validation'))
test_size = len(name_dict.item().get('test'))

# Batches per epoch. Also what goes in __len__() in generator2.py
train_steps = int(np.ceil(train_size/params['batch_size']))
val_steps = int(np.ceil(val_size/params['batch_size']))
test_steps = int(np.ceil(test_size/params['batch_size']))

print("Training events: ") 
print(train_size)
print("Validation events: ")
print(val_size)
print("Test events: ")
print(test_size)
print("About to enter DNNGenerator")

training_generator = DNNGenerator(train_steps, name_dict.item().get('train'), train_label_dict, 0, params['dim'], params['batch_size'], params['n_classes'], params['n_channels']) 
validation_generator = DNNGenerator(val_steps, name_dict.item().get('validation'), val_label_dict, 0, params['dim'], params['batch_size'], params['n_classes'], params['n_channels']) 
test_generator = DNNGenerator(test_steps, name_dict.item().get('test'), test_label_dict, 0, params['dim'], params['batch_size'], params['n_classes'], params['n_channels'])

# For easy reset of notebook state.
tf.compat.v1.keras.backend.clear_session()

config_proto = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True), intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, device_count = {'GPU': 1 , 'CPU': 1})

off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.compat.v1.Session(config=config_proto)
set_session(session)

# Design model
model = tf.compat.v1.keras.Sequential([
        tf.compat.v1.keras.layers.Conv2D(filters=args.conv, kernel_size=(3, 3), data_format="channels_last", activation=tf.compat.v1.nn.relu, input_shape=params['dim']),
        tf.compat.v1.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.compat.v1.keras.layers.Conv2D(filters=args.conv2, kernel_size=(3, 3), data_format="channels_last", activation=tf.compat.v1.nn.relu),
        tf.compat.v1.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.compat.v1.keras.layers.Flatten(),
        tf.compat.v1.keras.layers.Dropout(0.25),
        tf.compat.v1.keras.layers.Dense(args.dense, activation=tf.compat.v1.nn.relu),
        tf.compat.v1.keras.layers.Dropout(0.25),
        tf.compat.v1.keras.layers.Dense(args.dense2, activation=tf.compat.v1.nn.relu),
        tf.compat.v1.keras.layers.Dense(params['n_classes'], activation=tf.compat.v1.nn.softmax)
])

model.summary()
for layer in model.layers:
    print(layer)
    print(layer.input_shape)
    print(layer.output_shape)

# Training & validation settings
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['binary_accuracy'])

print("Will begin training the neural network model")

# Train model on already shuffled dataset
history = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=train_steps,  
                    epochs=args.epoch_num,
                    validation_steps=val_steps,
                    verbose=1, 
                    shuffle=False)

# Save trained model, weights, and other features
model.save('generator_2Dmodel.h5')

# Apply model on test set
for i in range(test_steps):
    images, labels_batch = test_generator[i]
    predict_batch = model.predict(images)
    predictions = np.concatenate((predictions, predict_batch), axis=0)
    test_labels = np.concatenate((test_labels, labels_batch), axis=0)

test_labels = np.argmax(test_labels, axis=1)
np.save("NN_predictions_" + args.job_num + ".npy", predictions[:,1])
np.save("NN_labels_" + args.job_num + ".npy", test_labels)

# Plot of training and validation accuracy vs epoch number
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Binary Accuracy')
plt.title('2D CNN plus channels')
plt.legend()

plt.savefig("binaryacc_vs_epoch_generator_" + args.job_num + ".png")
