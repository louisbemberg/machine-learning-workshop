# Controlling a Vehicle’s Steering Angle With Convolutional Neural Networks (CNN) and Udacity Simulator

This project develops a camera-based steering angle prediction with Keras and Udacity simulator.

本项目基于Udacity模拟器与Keras库训练了读取图像并预测转向角的卷积神经网络模型.

![](https://github.com/xiamze/steering_angle_prediction/blob/main/Image/1.png)

## Overview

Computer vision plays a key role as it processes images with rich and direct information in vehicle autonomy. Compared to technologies like radar, which provides only limited surrounding information through complicated processing, computer vision offers methods to detect and model every image element in a stable, efficient way at lower cost. Along with our Neural Network topics, our team has made an easy attempt to enable vehicle autonomy with CNN and udacity simulator. 

First, we generate a training dataset by manually controlling the car and images taken by its virtual cameras will be saved. Then we load the images and saved steering angles into a data augmentation pipeline to produce data diversity; augmented data will come into CNN training in batches, and finally we obtain a model that could predict the steering angle with an input image.

## Dependancies

Create an environment with [packages](https://github.com/xiamze/steering_angle_prediction/blob/main/environment.yml) (Anaconda recommended) by running the following command. If the code doesn't work, install manually by pip according to the .yml file.

```
conda env create -f environment.yml 
```
Remember to switch to the environment each time before use.
```
activate carnd-term1
```

## Training and Testing

Run the following to train the model.
```
python model.py
```

Open simulator.exe autonomous mode and run the following to load model into simulation.
```
python drive.py model.h5
```

## Credits
Simulator source: https://github.com/udacity/self-driving-car-sim

Helpful tutorial: https://github.com/HamdiTarek/Self_Driving_Car/tree/master/Self_Driving_Car



