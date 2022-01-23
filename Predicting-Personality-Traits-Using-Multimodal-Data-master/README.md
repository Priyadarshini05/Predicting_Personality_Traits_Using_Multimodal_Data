
# Predicting Personality Traits Using Multimodal Data
This project was done for Fachpraktikum Machine learning and Computer vision laboratory for Human Computer Interaction, University of Stuttgart

## Reference
1. Main paper to be cited ([Juan et al., 2016](http://www.eecs.qmul.ac.uk/mmv/datasets/amigos/doc/Paper_TAC.pdf))


```
@article{correa2018amigos,
  title={Amigos: A dataset for affect, personality and mood research on individuals and groups},
  author={Correa, Juan Abdon Miranda and Abadi, Mojtaba Khomami and Sebe, Niculae and Patras, Ioannis},
  journal={IEEE Transactions on Affective Computing},
  year={2018},
  publisher={IEEE}
  doi={10.1109/TAFFC.2018.2884461}
}
```
## Introduction

The aim of this project is to predict personality traits based on multimodal data using AMIGOS dataset consisting videos and biological signals like Electroencephalogram (EEG), Galvanic Skin Response (GSR) and Electrocardiogram (ECG). We firstly extract the gaze features from videos and physiological features from biological signals. Next, we apply machine learning algorithms like Random Forest (RF), Support Vector Machines (SVM) and eXtreme Gradient Boost (XGBoost) and different feature selection methods to personality recognition framework which performs binary classification of the Big-Five personality traits in the individual setting.

## Requirements

* Python 3
* BioSPPy
* PyEMD
* SciPy
* scikit-learn
* XGBoost
* OpenFace

## Dataset:

This is provided by [Juan et al., 2016].

## Overview:

Personality recognition frame-work

![Flow chart1](https://user-images.githubusercontent.com/73828269/109433243-a9fd9b00-7a0f-11eb-8ad7-1faa8821b993.png)


## Repo Usage:

For detailed explaination of how to use the repository, please refer the Project_How_To document


## Results:
RF trained on single modality (biological  signals) was compared  with  RF  trained  on  fusion  of  modalities (biological signals and gaze). As a result of feature level fusion of two modalities, F1-score of extraversion, agreeable-ness, conscientiousness, emotional stability was increasedby 0.076, 0.033, 0.309 and 0.128 respectively, compared to RF with single modality.

![RF bio vs fusion](https://user-images.githubusercontent.com/73828269/109493522-eb805b80-7a8c-11eb-8ff6-9fa0231fe186.png)




## Acknowledgement:
<body> <a> Reference Repository </a> <a href="https://github.com/pokang-liu/AMIGOS/blob/master/main.py"> -  Repo </body> <br>
<body> <a> Code for equation of circle </a> <a href="https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/"> -  Pupil Diameter </body> 


 

