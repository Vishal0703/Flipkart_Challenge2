# Training the model

We used keras-retinate module to train our model using transfer learning. The pretrained COCO weights were used and the training was done by freezing the 
backbone of the given architecture.

Feature selection was not a problem at all as these models use Deep CNN networks and ROI proposals.

# Data Augmentation

To train the model so as to make it robust to orientations we augmented our data with several rotated versions of the training images.
The subsampling, rescaling and all other preprocessing was already taken care of by the keras-retinanet module.