Hello!

Below you can find a outline of how to reproduce my solution for the RSNA Intracranial Hemorrhage Detection competition. </br>

Visit kaggle forum for solution overview: [Kaggle RSNA Intracranial Hemorrhage Detection: 4th Place Solution](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/discussion/118249#latest-678438)

Our code is based on Appian's repo: https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage

# HARDWARE

* Ubuntu 16.04
* NVIDIA 2080Ti

# SOFTWARE
(python packages are detailed separately in `requirements.txt`)
* Python 3.6.7
* CUDA 10.1
* CUDNN 7501
* NVIDIA Drivers 418.67

# START
1. Setup environment</br> 
2. Place the raw data into ./IFE_1/input folder.</br>  
	1. The test data correspond to the test data provided in the Stage 2 of competition. </br>  
	2. Use stage 2 training data to train the model.<br>  
3. cd IFE_1, run ./bin/preprocess.py to preprocess the training and test images and split the training data into five folds.</br> 
4. To train:</br> 
	1. Train feature extraction models</br>  
		* Go to IFE_1, IFE_2, IFE_3, run ./bin/train.sh to train five fold models. Models are saved in /model/. Best models are saved as foldx_best.pt.</br>   
		* It will take about 24  ~ 48 hours to train each model for one fold.</br> 
	2. Extract features</br>   
		* Go to IFE_1, IFE_2, IFE_3, run ./bin/gen_feat_train.sh and ./bin/gen_feat_test.sh to generate 1D (and 3D features). Use the best models generated from step 4.1.1.</br>   
		* It will take around 5 hours to extract one feature set (train/test TTA5).</br> 
	3. Train classification models.</br>   
		* Go to folder cls_1, cls_2, cls_3, run ./bin/train.sh, train five fold models for each folder.</br>  
		* It will take around 3 hours to train 1D+3D model (single model), and around 1.5 hours to train 1D model (single model). </br> 
5. To infer:</br> 
	1. Extract test features. </br> 
		* Go to folder IFE_1, IFE_2, IFE_3, run ./bin/gen_feat_test.sh to extract test features.</br>   
	2. Predict classification probabilities</br> 
		* Go to folder cls_1, cls_2, cls_3, run ./bin/predict.sh to predict result using extracted features.</br>   
	3. Ensemble</br> 
		* run ./libs/ensemble.sh to ensemble all the predictions.</br>   
6. Models and features are generated in sequence. If one follows the above mentioned steps in order, all the softlinks should be valid by the time they are referred. </br>   
