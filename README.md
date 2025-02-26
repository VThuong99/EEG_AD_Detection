# EEG-based Alzheimer's Disease Detection

This repository contains an AI-based pipeline for processing EEG signals to diagnose Alzheimer's disease. The project includes data preprocessing, feature extraction, model training, and evaluation using Leave-one Subject-out (LOSO) cross-validation techniques. 

## Project Overview
This project focuses on applying machine learning techniques to analyze EEG signals for Alzheimer's detection. The pipeline consists of:
- **Dataset:** The study is based on the **ds004504** dataset from OpenNeuro, which contains EEG recordings relevant to Alzheimer's diagnosis.
- **Data Loading:** Reading EEG datasets.
- **Feature Extraction:** Computing band power and other EEG-related features.
- **Model Training & Evaluation:** Using machine learning models with LOSO cross-validation.
- **Performance Metrics:** Assessing model accuracy and robustness.