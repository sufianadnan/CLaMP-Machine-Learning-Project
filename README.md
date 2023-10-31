# Malware Detector Using Machine Learning

Malware is a significant cyber threat, encompassing various malicious software that can cause information theft, espionage, ransomware attacks, and more. The ever-growing diversity of malware poses a substantial challenge for traditional anti-virus scanners, leaving millions of hosts vulnerable to attacks. In 2021, approximately 1,220.46 million malwares were examined (AV TEST, May 2021), highlighting the magnitude of this issue. Cybercrime, largely fueled by malware, is predicted to cost the world $10.5 trillion annually by 2025.

## Project Objective

The primary objective of this project is to develop a machine learning-based malware detector and evaluate its accuracy. We aim to conduct static analysis on malware samples, identify optimal features for different machine learning algorithms, and determine the most accurate algorithm for distinguishing malware from benign samples with the lowest error rate.

## Problem Statement

### Signature-Based vs. Behavior-Based Analysis

In this project, we explore two primary approaches for malware detection:

1. **Signature-Based Analysis:** This method involves comparing the digital signature of files with known malware signatures. It is effective in identifying known malware but is less useful for detecting zero-day vulnerabilities.

2. **Behavior-Based (Heuristic) Analysis:** This approach focuses on identifying malicious behavior exhibited by software. It is crucial for detecting previously unknown malware, especially zero-day threats.

In our assignment, we will elaborate on the theoretical basis of both signature-based and behavior-based analysis and discuss when and where to use each approach.

## Steps to Build the Malware Detector

### 1. Collect and Prepare the Data

We will begin by collecting malware samples for our dataset. You can download the dataset from "Classification of Malwares (CLaMP)." This dataset will serve as the foundation for our machine learning model.

### 2. Feature Extraction

To reduce the dimensionality of the dataset, we will perform feature selection. We aim to identify the minimum number of features for each type of analysis that will yield the best results.

### 3. Feature Transformation

After feature extraction, we will perform various data preprocessing steps, such as cleaning, normalization, and standardization. If necessary, we will also explore non-linear transformations to prepare the features for feeding into our machine learning model.

### 4. Train the Model

We will use at least two different supervised machine learning algorithms to train our malware detection model. The selected algorithms include:

1. J48 Decision Tree
2. Support Vector Machine (SVM)
3. K-Nearest Neighbors (KNN)

### 5. Cross Validation

Machine learning algorithms require cross-validation to assess their performance accurately. We will utilize three different cross-validation methods:

1. Holdout Method
2. K-Fold Method
3. Leave-One-Out Method

### 6. Improve Results

We will calculate accuracy and false positive rates for each of our models and aim to improve their performance. This might involve tuning hyperparameters, feature engineering, or trying different machine learning techniques.

### 7. Present Results

To present our findings, we will use graphs and statistical tools to compare the results obtained using different algorithms. We will identify the characteristics and use cases for each algorithm and provide insights into their success rates.

This README serves as an overview of our project on building a machine learning-based malware detector. You can find more detailed information in our project report. Feel free to explore the code and resources in the repository to understand our methodology and findings.

**Project Author:** Sufian Adnan

**Note:** Please ensure that all ethical and legal considerations are adhered to when working with malware samples.
