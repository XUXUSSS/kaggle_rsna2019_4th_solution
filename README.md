Hello!

Below you can find a outline of how to reproduce my solution for the RSNA Intracranial Hemorrhage Detection competition.

If you run into any trouble wit the setup/code or have any questions please contact me at mengdi.xu@gmail.com

#HARDWARE
Ubuntu 16.04
NVIDIA 2080TI

#SOFTWARE(python packages are detailed separately in `requirements.txt`):
Python 3.6.7
CUDA 10.1
CUDNN 7501
NVIDIA Drivers 418.67

a. Setup environment
b. Place the raw data into ./IFE_1/input folder.
	1. The test data correspond to the test data provided in the Stage 2 of competition. 
	2. Use stage 2 training data to train the model.
c. cd IFE_1, run ./bin/preprocess.py to preprocess the training and test images and split the training data into five folds.
d. To train:
	1. Train feature extraction models
		1. Go to IFE_1, IFE_2, IFE_3, run ./bin/train.sh to train five fold models. Models are saved in /model/. Best models are saved as foldx_best.pt.
	2. Extract features
		1. Go to IFE_1, IFE_2, IFE_3, run ./bin/gen_feat_train.sh and ./bin/gen_feat_test.sh to generate 1D (and 3D features). Use the best models generated from step d1.1. 
	3. Train classification models.
		1. Go to folder cls_1, cls_2, cls_3, run ./bin/train.sh, train five fold models for each folder.
e. To infer:
	1. Extract test features. 
		1. Go to folder, IFE_1, IFE_2, IFE_3, run ./bin/gen_feat_test.sh to extract test features.
	2. Predict classification probabilities
		1. Go to folder cls_1, cls_2, cls_3, run ./bin/predict.sh to predict result using extracted features.
	3. Merge the results
		1. run ./libs/ensemble.sh to average all the predictions.
f. Models and features are generated in sequence. If one follows the above mentioned steps in order, all the softlinks should be valid by the time they are referred. 
