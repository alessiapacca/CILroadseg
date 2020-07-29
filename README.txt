+========================================================+
|                                                        |
|              Satellite Road Segmentation               |
|             Computational Intelligence Lab             |
|                   ETH Zurich, FS2020                   |
|                                                        |
|    N. Kumar, L. Laneve, A. Paccagnella, T. Pegolotti   |
|                                                        |
+========================================================+

Here we summarize how to reproduce our code. In order to
understand the structure and usage of our classes, we
recommend to follow the Jupyter Notebooks we provided in
this directory.

==========================================================
REPRODUCIBLE EXPERIMENTS:
- zero_classifier.ipynb
  This notebook shows how we estimated, using a 4-fold
  cross validation, the accuracy of the zero classifier,
  as described in the report, Section II.

- logistic.ipynb
  This notebook shows how to train, use and visualize
  the results of the logistic regression baseline
  described in the report, Section II(A).

- naive_cnn.ipynb
  This notebook shows how to train, use and visualize
  the results of the context-free CNN baseline
  described in the report, Section II(B).

- cnn.ipynb
  This notebook shows how to train, use and visualize
  the results of the context-sensitive CNN model
  described in the report, Section III(A).

- xception.ipynb
  This notebook shows how to train, use and visualize
  the results of the Xception based classifier
  described in the report, Section III(B).

- unet.ipynb
  This notebook shows how to train, use and visualize
  the results of the U-Net model described in the
  report, Section IV.

- discontinuity.ipynb
  This notebook shows the visual differences between
  a U-Net prediction made of 4 or 9 pieces. This
  problem was mentioned in the report, Section IV.

- bagging.ipynb
  This notebook shows how we computed the majority of
  9 different (already trained) U-Nets, in order to
  obtain our best result, as mentioned in the report,
  Section VI.

==========================================================
UTILITY FILES:
- model_base.py
  Defines a base class with a common interface we used
  for all our models, in order to be able to define
  common functions.

- config.py
  Contains the definition of the functions for
  computing the metrics we need.

- cross_validation.py
  Defines a function which computes and prints a K-fold
  cross validation, estimating the metrics defined in
  config.py.

- helpers.py
  Defines function to load and save images.

- notebooks.py
  Defines utility functions useful to make notebooks
  more straightforward to read, like loading data,
  making predictions...

- submit.py
  Contains the implementation for CSV generation
  provided in the Kaggle competition.

- visualize.py
  Defines utility functions to visualize images
  on the notebooks.

==========================================================
OTHER DIRECTORIES:
- training/
  Contains the provided training set, and the additional
  training data extracted from the dataset cited in
  the report, Section III(B).

- test/images/
  Contains the provided test set.

- test/
  All the notebooks will save generated CSV files into
  this directory.

- test/pred/
  All the notebooks will save the predicted masks as
  .png files into this directory. The CSV files are
  then generated using these images.

- saves/final/
  All the .h5 files containing the weights of our
  neural networks are here.

==========================================================
WEIGHT FILES: [in saves/final/]
These weight files produce the results that, on the
Kaggle public test set, achieve the F score mentioned
in report, Section VIII.

Notice that rotate and mean (RM) does not change the
training and thus it does not have a different .h5 file.

- unet-base.h5:
        unet without any data augmentation,
        only bootstrapping of subimage
- unet-rotation.h5
        unet with only rotation/zoom/flip augmentation
- unet-rotation+color.h5
        unet with rotation/zoom/flip augmentation, as
        well as brightness/contrast augmentation.

- xception-base.h5
        xception with rotation/flip augmentation
- xception-additionaldata.h5
        xception with rotation/flip augmentation,
        trained with the additional data mentioned
        in the report, Section III(B).

==========================================================
VOTER PARAMETERS:
In saves/final/bagging/ we have 9 different weight files.
All of them have rotate/flip/zoom augmentation.

- unet-1.h5
        no color augmentation
- unet-2.h5
        brightness range [1, 1.1]
        contrast range [1, 1.1]
- unet-3.h5
        brightness range [1, 1.1]
        contrast range [1, 1.1]
- unet-4.h5
        brightness range [1, 1.1]
        contrast range [1, 1.1]
        dropout 0.1 on bottom layer (before first upsample)
- unet-5.h5
        brightness range [1, 1.2]
        no contrast
- unet-6.h5
        brightness range [1, 1.2]
        contrast range [1, 1.1]
- unet-7.h5
        brightness range [0.9, 1.2]
        contrast range [0.9, 1.2]
- unet-8.h5
        brightness range [1, 1.1]
        no contrast
        dropout 0.1 on bottom layer (before first upsample)
- unet-9.h5
        brightness range [1, 1.1]
        contrast range [1, 1.1]