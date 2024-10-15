# 02_01_start.py

# This file starts empty and we will gradually build on it in subsequent files.

import os
import numpy as np
import matplotlib.pylot as plt
import tensorflow.keros.datasets import cifar10
from tensorflow.keros.utils import to_categorical

#disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

#Ensure Tensorflow uses CPU only
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
tf.config.set_visible_devices([],'GPU')

#LOAD the dataset

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Normalization
X_train = X_train.astype('float32') /255
X_test = X_test.astype('float32') /255

#Convert class labels to one-hot encoded vectors

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#Define Labels of Dataset
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog','frog', 'horse', 'ship', 'truck']

#Print the shapes of the datasets to verify transformations
print(f"X_train shape: {X_train.shapa}")
print(f"y_train shape after one-hot encoding: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape after one-hot encoding: {y_test.shape}")

#Output Lubrary
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))

#plot directory
plot_path = os.path.join(output_dir, 'plots')

#Create the plot path if it does not exist
if not os-path.exists(plot_path):
  os.makedirs(plot_path)

# Function to display a sample of images from the dataset and save the plot
def display_images(images, labels, y_data, rows=4, cols=4, save_path=None):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.ravel()
    for i in np.arange(0, rows * cols):
        index = np.random.randint(0, len(images))
        axes[i].imshow(images[index])
        label_index = np.argmax(y_data[index])  # Get the index of the label
        axes[i].set_title(labels[label_index])
        axes[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    plt.show()  # Show the plot
    plt.close()  # Close the figure after showing it

# Define the file path to save the plot
plot_file = os.path.join(plot_path, 'display_images.png')

# Display a sample of training images with their labels and save the plot
display_images(X_train, labels, y_train, save_path=plot_file)