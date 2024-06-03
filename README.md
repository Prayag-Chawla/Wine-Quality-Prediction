
## Wine Quality Prediction
In this project, I have done wine quality prediction. I have first used the concept of eexploratory data analysis, and then I have used various methods of Machine Learning.

This dataset is related to red variants of the Portuguese "Vinho Verde" wine.The dataset describes the amount of various chemicals present in wine and their effect on it's quality. - The datasets can be viewed as classification or regression tasks.
This data frame contains the following columns:
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10- sulphates
11 - alcohol
12 - quality

## Work plan
EDA
Building a Machine Learning Model

## Linear regression

Linear regression is a fundamental statistical method used to model the relationship between a dependent variable and one or more independent variables. In simple linear regression, the relationship between one dependent variable and one independent variable is modeled with a straight line, represented by the equation 
The process involves collecting and preprocessing data, fitting the model using software like statsmodels or scikit-learn, checking assumptions (linearity, independence, homoscedasticity, normality, and no multicollinearity), and evaluating the model's performance with metrics such as R-squared and Mean Squared Error (MSE). Once validated, the model can be used for predictions, providing valuable insights and forecasting capabilities.







## Logistic regression

Logistic regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables, particularly when the dependent variable is binary (i.e., it has two possible outcomes such as "yes" or "no", "success" or "failure"). Unlike linear regression, which predicts continuous outcomes, logistic regression predicts the probability that a given input point belongs to a certain class.

In logistic regression, the output is transformed using a logistic function (also known as the sigmoid function), which produces a probability value between 0 and 1. This makes it suitable for classification problems. The model estimates the probability that a given observation belongs to the default class (usually coded as 1).

## Decision tree classifier

A decision tree classifier is a type of supervised machine learning algorithm used for classification tasks. It models decisions and their possible consequences as a tree-like structure of choices. Each internal node represents a "test" or decision on an attribute, each branch represents the outcome of that decision, and each leaf node represents a class label (the decision taken after computing all attributes).

Key Concepts
Nodes:

Root Node: The top node of the tree representing the entire dataset, which is then split into two or more homogeneous sets.
Decision Nodes: Nodes that split into further nodes based on certain conditions.
Leaf Nodes (Terminal Nodes): Nodes that do not split further and represent the final classification outcome.
Splitting: The process of dividing a node into two or more sub-nodes based on certain conditions or criteria.

Attribute Selection Measures: Metrics like Gini Index, Information Gain, or Chi-Square are used to select the attribute that best splits the data into distinct classes.

Pruning: The process of removing parts of the tree that do not provide additional power to classify instances, aimed at reducing overfitting and improving the model's generalization.

Steps in Building a Decision Tree Classifier
Data Collection: Gather the data with the target variable and predictor variables.
Data Preprocessing: Clean the data, handle missing values, and convert categorical variables into numerical values if necessary.
Choosing the Best Attribute: Use attribute selection measures to determine the best attribute to split the data.
Splitting: Divide the data into subsets based on the best attribute, creating branches of the tree.
Repeat Steps 3-4: Continue splitting each subset recursively, using the best attribute at each step, until all data is classified or a stopping criterion (like maximum depth of the tree) is met.
Pruning: Remove unnecessary branches from the tree to avoid overfitting and improve the tree's performance on unseen data.

## Support vector machine

Support Vector Machine (SVM) is a robust supervised machine learning algorithm primarily used for classification tasks but also applicable to regression. It works by finding the optimal hyperplane that separates data points of different classes with the maximum margin. The closest data points to this hyperplane are called support vectors, which are crucial in defining the boundary. SVM is particularly effective in high-dimensional spaces and can handle both linear and non-linear classification problems through the use of kernel functions, such as linear, polynomial, and radial basis function (RBF) kernels. Despite its strengths, including robustness to overfitting and versatility, SVM can be computationally intensive, especially with large datasets, and selecting the appropriate kernel and tuning its parameters can be complex. SVMs are widely used in various fields like bioinformatics for gene classification, text categorization, and image recognition, due to their accuracy and effectiveness in handling complex datasets.



## Catboost Regressor
CatBoost is a gradient boosting algorithm developed by Yandex, optimized for handling categorical data efficiently and effectively. As a regression tool, CatBoost builds an ensemble of trees where each new tree aims to correct the errors of the previous ones, ensuring robust predictive performance. It uniquely processes categorical features directly, eliminating the need for extensive preprocessing like one-hot encoding, which can lead to better performance and faster training times. CatBoost is user-friendly, requiring minimal parameter tuning, and includes built-in mechanisms to prevent overfitting, such as ordered boosting and advanced regularization techniques. Despite being resource-intensive, CatBoost is particularly effective in fields like finance, marketing, and healthcare, where high-dimensional categorical data is prevalent, making it a popular choice for complex regression tasks.

## Libraries and Usage

```
#IMPORTING ALL THE LIBRARIES

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")  #using style ggplot

%matplotlib inline

import plotly.graph_objects as go
import plotly.express as px
```






## Accuracy
These models were selected and evaluated:¶
linear_model
1- LinearRegression
Score the X-train with Y-train is : 0.37%
Score the X-test with Y-test is : 0.34%
Mean absolute error is 0.47%
Mean squared error is 0.37%
Median absolute error is 0.37%

2- LogisticRegression

Score the X-train with Y-train is : 0.58%

Score the X-test with Y-test is : 0.62%
Mean absolute error is 0.40%
Mean squared error is 0.45%
Median absolute error is 0.0
Accuracy score 0.625%
Decision Tree Classifier
Score the X-train with Y-train is : 0.92%
Score the X-test with Y-test is : 0.54%
Accuracy score 0.54%
Model SVM
1- SVC
Score the X-train with Y-train is : 0.59%
Score the X-test with Y-test is : 0.64%
Accuracy score 0.64%

2-SVR

Score the X-train with Y-train is : 0.14%

Score the X-test with Y-test is : 0.23%
Accuracy score 0.64%
Neighbors model
Score the X-train with Y-train is : 0.62%
Score the X-test with Y-test is : 0.53%
Accuracy score 0.53%
CatBoost model
Score the X-train with Y-train is : 1%
Score the X-test with Y-test is : 0.65%
Accuracy score 0.53%
ExtraTrees model
Score the X-train with Y-train is : 1%
Score the X-test with Y-test is : 0.58%
Accuracy score 0.53%
Stacking model
Score the X-train with Y-train is : 0.94%
Score the X-test with Y-test is : 0.68%
Accuracy score 0.53%





## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## Used By
In the real world, this project is used infood and drinks industry extensively.
## Appendix

A very crucial project in the realm of data science and new age predictions domain using visualization techniques as well as machine learning modelling.

## Tech Stack

**Client:** Python, Naive byes classifier, gaussian naive byes, suppport vector machine, stack model, linear regression, decision tree classifier, logistic regression model, EDA analysis, machine learning, sequential model of ML,, data visualization libraries of python.



## Feedback

If you have any feedback, please reach out to us at chawlapc.619@gmail.com

