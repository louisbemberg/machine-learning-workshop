## =============================================================
## TOPIC: STEERING ANGLE PREDICTION WITH KERAS/UDACITY
## Overview: Train a camera-based steering angle predictor
## in Keras CNN and validate via Udacity simulator
## =============================================================

## Import packages needed for image processing/CNN
import csv
import cv2
import numpy as np
from matplotlib import pyplot
from livelossplot import PlotLossesKeras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Lambda, Cropping2D, Dropout

## Flip horizontally with image and angle to add data diversity
## since simulation map 1 has way more left turn than right
## turn, do so also prevents the tendency to turn left
def flip(image, angle):
	## 50% chance to flip
	choice = np.random.choice([0, 1])
	if choice:
		image, angle = cv2.flip(image, 1), -angle
	return (image, angle)

## Randomly decrease brightness for model robustness
## research reveals NN predict is impacted by brightness
## change, do so to adjust the model to different occasion
def brightness(image, angle):
	## HSV's V(value) means brightness, easy to adjust
	## so we switch to HSV range and switch back after change
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	alpha = np.random.uniform(low=0.1, high=1.0, size=None)
	v = hsv[:, :, 2]
	v = v * alpha
	hsv[:, :, 2] = v.astype('uint8')
	rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
	return (rgb, angle)

## Randomly discard data with 0 angle
## nonlinearity of the simulator steering brings a majority of
## 0-value data, which destroys balance between data types
def discard(angles, rate):
	## Return index of randomly selected 0 value for 'np.delete'
	steering_zero_idx = np.where(angles == 0)
	steering_zero_idx = steering_zero_idx[0]
	## Proportation of data to be discarded
	size = int(len(steering_zero_idx) * rate)
	return np.random.choice(steering_zero_idx, size=size, replace=False)

## Intergrated previous functions
def transform(image, angle):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image, angle = flip(image, angle)
	image, angle = brightness(image, angle)
	return (image, angle)

def keras(shape):
    model = Sequential()
    ## Cropping layer to cut useless pixels (trees/sky/hood etc.)
    model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=shape))
    ## Regularization down to (-0.5, 0.5)
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    
    ## Convolution begins
    ## Return a single float as steering angle
    model.add(Convolution2D(8, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(8, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(16, (4, 4), strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(16, (5, 5), strides=(1, 1), padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))     
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.25))    
    model.add(Dense(1, activation='linear'))
    
    return model


if __name__ == '__main__':

	## Read the driving log and extract data by lines
	lines = []
	with open('your_cvs_file_name.csv') as csvfile:
	    reader = csv.reader(csvfile)
	    for line in reader:
	        lines.append(line)
	
	## Lists to store original data
	images, angles = [], []

	## Extract picture name and angle from log lines
	for line in lines:
	    source_path = line[0]
	    filename = source_path.split("\\")[-1]
	    ## Create image path
	    curr_path = 'your_picture_folder_name\\' + filename
	    ## Save all image into images
	    image = cv2.imread(curr_path)
	    images.append(image)
	    ## Save the corresponding steering angle into measurements
	    angle = float(line[3])
	    angles.append(angle)

	## New lists to store original data + augmentated data
	x_, y_ = [], []
	for image, angle in zip(images, angles):
		x_.append(image)
		y_.append(angle)
		aug_image, aug_angle = transform(image, angle)
		x_.append(aug_image)
		y_.append(aug_angle)

	x_ = np.array(x_)
	y_ = np.array(y_)

	## Model parameters preset
	discard_rate = 0.5
	shape = (160, 320, 3)
	epochs = 30

	## Discard zero values
	discard_idx = discard(y_, discard_rate)
	X_train = np.delete(x_, discard_idx, axis=0)
	y_train = np.delete(y_, discard_idx, axis=0)

	## Model build and training
	model = keras(shape)
	print(model.summary())

	model.compile(loss='mse', optimizer='adam')
	history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs)

	## Save loss values as jpg
	pyplot.plot(history.history['loss'])
	pyplot.plot(history.history['val_loss'])
	pyplot.title('model train vs validation loss')
	pyplot.ylabel('loss')
	pyplot.xlabel('epoch')
	pyplot.legend(['train', 'validation'], loc='upper right')
	pyplot.savefig('train_val_loss.jpg')
	
	## Save model to be tested on the Udacity simulator
	model.save('model.h5')