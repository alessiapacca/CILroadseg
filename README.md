# Satellite Road Segmentation with Neural Networks - Computational Intelligence Lab <br>            
<p align="center">
  ETH Zurich, FS2020 <br>
  N. Kumar, L. Laneve, A. Paccagnella, T. Pegolotti
</p>    

<p align="center">
  <img src="https://github.com/alessiapacca/CILroadseg/blob/master/example.png" width="650" title="example of satellite image with his groundtruth segmentation mask">
</p>

The task of the project was to create a classification model able to segment aerial satellite images from ***Google Maps***, separating roads from backgrounds. We present and compare two different methods to tackle this problem, both based on ***Convolutional Neural Networks***: one based on a per-patch classification, and the other based on a fully convolutional network. The best model obtained in Kaggle’s public test data achieved an F1 score of ```0.92105```. <br>
For details about our implementation, please read the file ```report.pdf```. <br>
For details about how to reproduce our code, please read the file ```README.txt```.<br><br>

                                                        
**REPRODUCIBLE EXPERIMENTS**: 
* ```zero_classifier.ipynb``` <br>
  This notebook shows how we estimated, using a 4-fold
  cross validation, the accuracy of the zero classifier,
  as described in the report paper, Section II.

* ```logistic.ipynb``` <br>
  This notebook shows how to train, use and visualize
  the results of the logistic regression baseline
  described in the report paper, Section II(A).

* ```naive_cnn.ipynb``` <br>
  This notebook shows how to train, use and visualize
  the results of the context-free CNN baseline
  described in the report paper, Section II(B).

* ```cnn.ipynb``` <br>
  This notebook shows how to train, use and visualize
  the results of the context-sensitive CNN model
  described in the report paper, Section III(A).

* ```xception.ipynb``` <br>
  This notebook shows how to train, use and visualize
  the results of the Xception based classifier
  described in the report paper, Section III(B).

* ```unet.ipynb``` <br>
  This notebook shows how to train, use and visualize
  the results of the U-Net model described in the
  report paper, Section IV.

* ```discontinuity.ipynb``` <br>
  This notebook shows the visual differences between
  a U-Net prediction made of 4 or 9 pieces. This
  problem was mentioned in the report paper, Section IV.

* ```bagging.ipynb``` <br>
  This notebook shows how we computed the majority of
  9 different (already trained) U-Nets, in order to
  obtain our best result, as mentioned in the report paper,
  Section VI.
