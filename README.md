# Predicting_Personality_Traits_Using_Multimodal_Data
This project was done for Fachpraktikum Machine learning and Computer vision laboratory for Human Computer Interaction, University of Stuttgart

# Reference
[Juan et al., 2016](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/doc/Paper_TAC.pdf)
@article{correa2018amigos,<br>
  title={Amigos: A dataset for affect, personality and mood research on individuals and groups},<br>
  author={Correa, Juan Abdon Miranda and Abadi, Mojtaba Khomami and Sebe, Niculae and Patras, Ioannis},<br>
  journal={IEEE Transactions on Affective Computing},<br>
  year={2018},<br>
  publisher={IEEE}<br>
  doi={10.1109/TAFFC.2018.2884461}<br>
}

# Introduction

The aim of this project is to predict personality traits based on multimodal data using AMIGOS dataset consisting videos and biological signals like Electroencephalogram (EEG), Galvanic Skin Response (GSR) and Electrocardiogram (ECG). We firstly extract the gaze features from videos and physiological features from biological signals. Next, we apply machine learning algorithms like Random Forest (RF), Support Vector Machines (SVM) and eXtreme Gradient Boost (XGBoost) and different feature selection methods to personality recognition framework which performs binary classification of the Big-Five personality traits in the individual setting.

# Requirements
Python 3<br>
BioSPPy<br>
PyEMD<br>
SciPy<br>
scikit-learn<br>
XGBoost<br>
OpenFace<br>

# Dataset
This is provided by [Juan et al., 2016](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/doc/Paper_TAC.pdf)

# Overview
![image](https://user-images.githubusercontent.com/43397172/150693549-c115f064-2948-4d7e-92c4-859665fa44d1.png)

# Repo Usage
For detailed explaination of how to use the repository, please refer the Project_How_To document

# Results
RF trained on single modality (biological signals) was compared with RF trained on fusion of modalities (biological signals and gaze). As a result of feature level fusion of two modalities, F1-score of extraversion, agreeable-ness, conscientiousness, emotional stability was increasedby 0.076, 0.033, 0.309 and 0.128 respectively, compared to RF with single modality.

![image](https://user-images.githubusercontent.com/43397172/150693776-bf867cfe-84c3-48d7-8e35-ed55c87a86f4.png)

# Acknowledgement

Reference Repo: [Repo](https://github.com/pokang-liu/AMIGOS/blob/master/main.py)<br>
Pupil diameter equation: [Link](https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/)



