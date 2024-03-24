# Project Objective

This repo is the neural network that is to be used with robot-sim. It uses an open socket to communicate with Unity to recieve image and position etc. data and returns control outputs to Unity.

# Dependencies

This repo uses Keras Tensorflow and Sockets.

# Command Line

## To train:

python train.py

# Looks for recordings made by the Unity game in directory "../robot_sim/Assets/Recordings/", so you should have robot_sim have the same parent directory as this repo.

## To run:

python infer.py

# Troubleshooting

- infer.py
	- Make sure you have already run train.py and that it created a model in the Saved Model directory.
	- The game needs to be running, and (at the moment) both programs (robot_sim game and train.py) need to be run on the same machine.
- train.py
	- Make sure you have the robot-sim repo installed. Its folder (robot_sim) should be placed in the same folder that this repo's folder is.