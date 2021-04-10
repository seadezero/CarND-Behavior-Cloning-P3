from math import ceil
from random import shuffle
import numpy as np
import csv
import cv2
import sklearn

CORRECT_FACTOR = 0.6
BATCH_SIZE = 32
EPOCHS = 10

def add_image(img_path, measurement, images, measurements):
  image = cv2.imread(img_path)
  images.append(image)
  measurements.append(measurement)

def generator(samples, batch_size=BATCH_SIZE):
  num_samples = len(samples)
  while 1: # Loop forever so the generator never terminates
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]

      images = []
      measurements = []
      for line in batch_samples:
        # Add center image
#         add_image(line[0], float(line[3]), images, measurements)
        center_image = cv2.imread(line[0])
        images.append(center_image)
        measurements.append(float(line[3]))

        # Add left image
#         add_image(line[1], float(line[3])+CORRECT_FACTOR, images, measurements)
        left_image = cv2.imread(line[1])
        images.append(left_image)
        measurements.append(float(line[3])+CORRECT_FACTOR)

        # Add right image
#         add_image(line[2], float(line[3])-CORRECT_FACTOR, images, measurements)
        right_image = cv2.imread(line[2])
        images.append(right_image)
        measurements.append(float(line[3])-CORRECT_FACTOR)

      # trim image to only see section with road
      X_train = np.array(images)
      y_train = np.array(measurements)
      yield sklearn.utils.shuffle(X_train, y_train)

lines = []

# Reads lines from normal runs
with open('norm-loop/1/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  next(reader)
  for line in reader:
    lines.append(line)
with open('norm-loop/2/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  next(reader)
  for line in reader:
    lines.append(line)

# Reads lines from reverse runs
with open('reverse-run/1/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  next(reader)
  for line in reader:
    lines.append(line)  
    
# Reads lines from recovery data
with open('recovery-data/1/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  next(reader)
  for line in reader:
    lines.append(line)  
    
print(len(lines))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Manually builds a convolutional neural network from ground up
model = Sequential()

model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(6, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Conv2D(12, (3, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Conv2D(32, (5, 5)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Conv2D(64, (5, 5)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(8))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)/BATCH_SIZE),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)/BATCH_SIZE),
            epochs=EPOCHS, verbose=1)

model.save('model.h5')
