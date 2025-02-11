# Machine Learning Project: Protein Classification

## Overview

This project focuses on classifying proteins into positive and negative labels based on their features. The dataset consists of protein features and labels, with the majority of the data being unlabeled. The project explores various machine learning techniques to handle the challenge of learning from partially labeled data, specifically in the context of Positive-Unlabeled (PU) Learning.

## Dataset

The dataset is divided into three main files:

- **training_others_features.csv**: Contains features of proteins that are not labeled.
- **training_pos_features.csv**: Contains features of proteins that are positively labeled.
- **training_pos_labels.csv**: Contains the labels for the positively labeled proteins.

### Dataset Statistics

- **Total Proteins**: 207,747
- **Positive Proteins**: 10% of the total
- **Negative Proteins**: 90% of the total

## Project Structure

The project is structured as follows:

### 1. Data Loading and Preliminary Analysis:
- Load the datasets.
- Perform initial data exploration and analysis.
- Visualize the distribution of positive and negative labels.

### 2. Feature Selection and Preprocessing:
- Standardize the features using `StandardScaler`.
- Apply variance thresholding to remove low-variance features.

### 3. Model Training and Evaluation:
- Split the data into training and testing sets.
- Train a `RandomForestClassifier`.
- Evaluate the model using metrics such as **ROC-AUC**, **F1-Score**, and **Classification Report**.
- Generate learning curves to assess model performance.

### 4. Handling Unlabeled Data:
- Explore **PU Learning** techniques to leverage the unlabeled data for improving model performance.

## Key Metrics

Given the imbalanced nature of the dataset, the following metrics are used to evaluate the model:

- **AUC-ROC**: Measures the model's ability to distinguish between positive and negative classes.
- **Precision and Recall**: Evaluate the model's performance in terms of false positives and false negatives.
- **F1-Score**: Balances precision and recall, providing a single metric for model evaluation.

## Challenges

- **Imbalanced Dataset**: Only 10% of the proteins are labeled, making it challenging to train a robust model.
- **PU Learning**: The project addresses the challenge of learning from partially labeled data, where only positive examples are labeled, and the rest are unlabeled.

## Dependencies

The project uses the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Usage

1. **Install Dependencies**:  
   Ensure you have the required libraries installed. You can install them using pip:
   
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn


2. **Run the Notebook**:
Open the ML_Project.ipynb notebook in Jupyter or any compatible environment and execute the cells to perform the analysis and model training.

3. **Explore Results**:
The notebook includes visualizations and model evaluation metrics to help understand the performance of the classifier.

## Conclusion

This project demonstrates the application of machine learning techniques to classify proteins based on their features, even when the majority of the data is unlabeled. By leveraging PU Learning and appropriate evaluation metrics, the project aims to build a robust classifier that can effectively distinguish between positive and negative protein labels.
Future Work

  * **Advanced PU Learning Techniques**: Explore more sophisticated PU Learning methods to further improve model performance.
  * **Feature Engineering**: Investigate additional feature engineering techniques to enhance the model's predictive power.
  * **Hyperparameter Tuning**: Perform hyperparameter optimization to fine-tune the model.
