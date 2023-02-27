# Clustering of image features of unknown objects and their sorting by means of a roboter

## General Information
- This project has its origin  from a batchelor thesis assignment which was worked out by Prof. Dr.-Ing. Nicolaj Stache.
- The supervisor of the thesis is M.Eng. Pascal Graf.
- This program code was developed by Dominic Doetterer(Mat-Nr.201974).


## What is this project about?
With this project unknown objects should be sorted automatically. The basis for decision-making is the clustering of data by using unsupervised learning.
The processing of the data is carried out with by a neural network.
####
The objects are guided over the conveyor belt and recorded by a camera. The collected data is then processed and sorted.
***
![](attachments/Aufbau_Gesamt_2Persp.png)
***


## Add libraries

```
pip install pandas
pip install pyueye
pip install opencv-python==4.5.3.56
pip install tensorflow
pip install serial
pip install matplotlib
pip install imutils
pip install plotly
pip install sklearn
pip install yellowbrick
pip install pyusb
```
## Hardware setup
- [Dobot Magician](https://variobotic.de/robotik-in-schulen/dobot-magician/?gclid=CjwKCAiAheacBhB8EiwAItVO2ztoIaly9RQJX57fD7foqoCuqpkj6LrmyVUgsiuRwS3cxY4sgQdq7xoC_78QAvD_BwE)
- [IDS UI5250CP-C-HQ](https://de.ids-imaging.com/download-details/AB00341.html)
- Conveyor belt
- Separation system

### Calibrate the system
There are two  calibrations that have to been done, before the sorting can be started.

1. The calibration bool in the main.py funktion have to mbe changed to 'TRUE'
2. Start the main.py

#### Threshold
The first calibration mode is to adjust the threshold for the image processing. The Sliders can be used to change the 
threshold. On the window is a live view of the current threshold boundaries.
The best way to find the optimal threshold is by pacing several items under the camera and adjust the setting so that
the most items get detected. By pressing 'q' the current threshold boundaries get saved.
###
#### Coordinates
The second calibration mode is required to calculate the x- and y-coordinates of the roboter correct.
Without the right calibration its imposible to calculate the coordinates out of the x-and y- camera coordinates.
The robot have to be on its final position and connected to the DobotStudio Software. If the setup is stationary, the calibration have to be done once.
####
The screen shows a rectangle. Two items have to be placed on the left upper and the right lower corner.
***
![](data/Calibrate/Calibrate.jpg)
***
If the objects are placed as shown the next step can be initialized by pressing 'q'.
Now the Dobot have to be maneuvered manually to the first item in the right lower corner. All Instructions are also 
visible on the console.
####
1. Input the x Roboter coordinates of Item #1
   - Press 'enter' to confirm
2. Input the y Roboter coordinates of Item #1
   - Press 'enter' to confirm
   - maneuvered manually to Position 2
3. Input the x Roboter coordinates of Item #2
   - Press 'enter' to confirm
4. Input the y Roboter coordinates of Item #2
   - Press 'enter' to confirm

Finally, check the z-height of the conveyor with the roboter and compare it to 'DoBot_belt_z' in the [Setup.py](Setup.py)

## How to operate the setup?
First connect the conveyor and the Dobot via USB with the computer. Then also connect the IDS camera via lan cable to the 
lan split, which provides the POE. 
Check the availability of the camera with the IDS camera manager and if necessary carry out a network configuration if the camera is not available. 
The computer obviously have to be connected to the same lan network.
####
After execute the main.py without errors, the GUI boots up. After some time the Roboter starts to home automatically.
The boot is done as soon the roboter is on its home position. Then all functions of the GUI can be used.
![](attachments/GUI_Main.png)
![](attachments/GUI_Equipment.png)

Each process can be interrupted by hitting the Stop button. 

## Data
The neural network needs a dataset of many pictures. Therefor the 256 Caltech dataset from Kaggle was used.
To train a Network there are the following steps to take:
1. Download Dataset "256 Caltech" from kaggle
2. Load Dataset into data/Neural Network/TrainingsData
####
 An already trained model is uploaded and used by default within this version. 
- data/Neural Network/Model.pkl


## Attachments
The attachments containing an CAD rendered picture of the setup. The .ino File for the Arduino of the conveyor is 
additionally deposited.



