# CSCI 6364 Machine Learning Project

## Team Members
* Xi Cheng
* Jing Si
* Mengqi Xie

## Introduction
This project is based on the Online News Popularity Data Set retrieved from UCI Machine Learning Repository, URL: <http://http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity/>

The goal is to train the learning algorithm to predict the popularity of a certain news. This will be  both a regression task to predict the exact number of shares and also classification for the level of popularity.

## Dataset Features
The dataset contains 39797 instances and 61 attributes. 

* 58 predictive attributes
* 2 non-predictive          
* 1 goal field

The popularity/unpopularity target is measured by the goal field which is the number of shares for the news.

## Project Workflow

1. Data wrangling and preprocessing
Divide the dataset into training set, validation set and test set and train the model;
The target of the dataset is the number of shared times. By evaluating the correlations of the 58 attributes, we should decide the most significant ones to join the model. 

2. Design Machine Learning Algorithms
Probable algorithm might be: Random Forest, SVM, or Naïve Bayes

3. Model Generation
Test different algorithms and find the best fitting one to generate models;
Validate these models(avoid overfitting etc.) and generated and choose the one with best fitting value.

4. Model Evaluation 
By applying the test dataset on the model, the result target value should give an acceptable  precise prediction.

## Reference
K. Fernandes, P. Vinagre and P. Cortez. A Proactive Intelligent Decision
    Support System for Predicting the Popularity of Online News. Proceedings
    of the 17th EPIA 2015 - Portuguese Conference on Artificial Intelligence,
    September, Coimbra, Portugal.

