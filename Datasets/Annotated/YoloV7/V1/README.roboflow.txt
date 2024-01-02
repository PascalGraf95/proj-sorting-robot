
Screw Teest - v2 2023-11-16 4:16pm
==============================

This dataset was exported via roboflow.com on November 16, 2023 at 3:16 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 44 images.
Screw-nut-washer are annotated in YOLO v3 (Keras) format.

The following pre-processing was applied to each image:
* Resize to 640x640 (Fit (black edges))

The following augmentation was applied to create 5 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Randomly crop between 0 and 17 percent of the image
* Random Gaussian blur of between 0 and 0.5 pixels
* Salt and pepper noise was applied to 1 percent of pixels


