#import capturetools
import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import re
import math

import tensorflow as tf
import torchvision.transforms as transforms

from PIL import Image

from tensorflow.keras import datasets, layers, models

from tensorflow.keras.applications import EfficientNetB0


batch_size = 3
epochs = 3 #was 125
learning_rate=0.00003

width = 512
height = 512

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

recordingsDir = "../robot_sim/Assets/Recordings/"

valid_ratio = 0.2

#Make list of file names
filenames = []
#List of validation file names
valid_filenames = []

for folder in os.listdir(recordingsDir):

    subfiles = []
    
    if os.path.isdir(recordingsDir+folder):

        for file in os.listdir(recordingsDir+folder):
            
            
            if file.endswith(".txt"):
                subfiles.append(file[:-4])
            
        
        subfiles.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        for f in subfiles:
            filenames.append(folder+'/'+f)
            
valid_filenames = filenames[int((1.0-valid_ratio)*len(filenames)):]
filenames = filenames[:int((1.0-valid_ratio)*len(filenames))]

#print(filenames)


def ReadImage(dir):
    
    #print(dir)
    
    image = tf.io.read_file(dir)
    image = tf.io.decode_png(image)
    image = tf.cast(image, tf.float32) / 255.0
    
    return np.asarray(image)

metaData = np.zeros(6)

def ReadMetaData(dir):
    metaData = [0.0,0.0,0.0,0.0,0.0,0.0]
    
    firstLine = True
    
    with open(recordingsDir+dir+".txt", newline='') as inputsFile:
        cReader = csv.reader(inputsFile, delimiter=',')
        for row in cReader:
            i = 0
            if not firstLine:
                metaData[0] = float(row[3])
                metaData[1] = float(row[4])
                metaData[2] = float(row[5])
                metaData[3] = float(row[6])
                metaData[4] = float(row[7])
                metaData[5] = float(row[8])
            firstLine = False
    
    inputsFile.close()
    
    return np.asarray(metaData)
    
    
def ReadLabel(dir):
    label = [0.0,0.0,0.0]
    
    firstLine = True
    
    with open(recordingsDir+dir+".txt", newline='') as inputsFile:
        cReader = csv.reader(inputsFile, delimiter=',')
        for row in cReader:
            i = 0
            if not firstLine:
                label[0] = float(row[0])
                label[1] = float(row[1])
                label[2] = float(row[2])
            firstLine = False
    
    inputsFile.close()
    
    return np.asarray(label)

#Generator class

class Generator(tf.keras.utils.Sequence):

    def __init__(self, dir_filenames, dir_valid_filenames, batch_size) :
        self.dir_filenames = dir_filenames
        self.dir_valid_filenames = dir_valid_filenames
        #self.labels = labels
        self.batch_size = batch_size
    
    
    def __len__(self) :
        return (np.ceil(len(self.dir_filenames) / float(self.batch_size))).astype(int)
  
  
    def __getitem__(self, idx, isValidation) :
    
        if(isValidation):
            batch_x = self.dir_valid_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
            mD = np.asarray([ReadMetaData(filename) for filename in batch_x])
            return np.array([
                np.resize(ReadImage(recordingsDir+filename+".png"), (width, height, 3))
                    for filename in batch_x]), mD, np.array([ReadLabel(filename) for filename in batch_x])
            
    
        batch_x = self.dir_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        #batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]
        
        mD = np.asarray([ReadMetaData(filename) for filename in batch_x])
        #print("metdat")
        #print(mD)
        
        return np.array([
            np.resize(ReadImage(recordingsDir+filename+".png"), (width, height, 3))
                for filename in batch_x]), mD, np.array([ReadLabel(filename) for filename in batch_x])



def MakeRecordingDataset(folder):

    images = []
    labels = []
    
    txtFiles = []

    for file in os.listdir(recordingsDir+folder):
        if file.endswith(".txt"):
            txtFiles.append(file)
    
    txtFiles.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    for file in txtFiles:
        imgname = file[:-4]+".png"
        
        image = ReadImage(recordingsDir+folder+"/"+imgname)
        
        label = ReadLabel(recordingsDir+folder+"/"+file)
        
        label_tensor = tf.convert_to_tensor(label)
        images.append(image)
        labels.append(label_tensor)
            
    images_tensor = tf.convert_to_tensor(images)
    labels_tensor = tf.convert_to_tensor(labels)
    
    return np.asarray(images), np.asarray(labels)
    #return images, labels
    

def MakeDataset():
    
    images = []
    labels = []
    
    for file in os.listdir("Recordings/"):
        ti, tl = MakeRecordingDataset(file)
        images.append(ti)
        labels.append(tl)
    
    
    il = []
    ll = []
    for lbl in labels:
        for entry in lbl:
            ll.append(entry)
    
    for img in images:
        for entry in img:
            il.append(entry)
    return np.asarray(il), np.asarray(ll)

train_generator = Generator(filenames, valid_filenames, batch_size)

class RoverModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.input1 = layers.Input(shape=(6))
        self.input2 = layers.Input(shape=(height, width, 3))
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.flat1 = layers.Flatten()
        self.dense1 = layers.Dense(128, activation='relu')
        self.reshape1 = layers.Reshape((-1, 32))
        self.lstm1 = layers.LSTM(32)
        #print(lstm1)
        self.concat = layers.Concatenate()
        self.dense2 = layers.Dense(32, activation='relu')
        self.output1 = layers.Dense(3, activation='relu')
    
    def call(self, in1, in2, training=False):
        x1 = in1
        x2 = in2
        x2 = self.conv1(x2)
        x2 = self.pool1(x2)
        x2 = self.conv2(x2)
        x2 = self.flat1(x2)
        x2 = self.dense1(x2)
        x2 = self.reshape1(x2)
        x2 = self.lstm1(x2)
        #print(x1.shape)
        #print(x2.shape)
        _x = self.concat([x1,x2])
        
        #_x = x2
        
        _x = self.dense2(_x)
        _x = self.output1(_x)
        
        return _x
        

model = RoverModel()

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

i = 0

epochCheckpoint = 1

#Temporary accuracy variables, at the moment they use only the training data as a metric
accScore = 0

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    
    bx2, bx1, blabel = train_generator.__getitem__(i, False)
    i += 1
    if i >= len(filenames) - batch_size - 1:
        i = 0
    
    x2 = bx2
    x1 = np.asarray(bx1)
    label = blabel
    
    with tf.GradientTape() as tape:
        # Run the forward pass of the layer.
        # The operations that the layer applies
        # to its inputs are going to be recorded
        # on the GradientTape.
        
        
        
        logits = model(x1, x2, training=True)  # Logits for this minibatch
        #print(logits)
        
        accs = np.zeros((label.shape[0],3))
        for u in range(label.shape[0]):
            accs[u,0] += 1.0/(abs(label[u,0] - logits[u,0])+10)
            accs[u,1] += 1.0/(abs(label[u,1] - logits[u,1])+10)
            accs[u,2] += 1.0/(abs(label[u,2] - logits[u,2])+10)
        
        for u in range(label.shape[0]):
            accScore += accs[u,0]
            accScore += accs[u,1]
            accScore += accs[u,2]

        # Compute the loss value for this minibatch.
        loss_value = loss_fn(label, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_value, model.trainable_weights)

    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    
    
    
    if epoch % epochCheckpoint == 0:
        
        acc = accScore / epochCheckpoint
        accScore = 0
        
        print(
            "Training loss (for one batch) at step %d: %.4f"
            % (epoch, float(loss_value))
        )
        print("Accuracy: " + str(acc))
        #print("Seen so far: %s samples" % ((epoch + 1) * batch_size))


model.save("Saved Model/")