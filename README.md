# Project Proposal

## Introduction

Businesses provide numerous services to their clients. It becomes important for them to know whether their clients need those services or not. Banking is one such sector which provides numerous services to their clients. If a client subscribes to their service, it increases revenue for the bank. One such service offered by banks is *Term Deposit*. A term deposit is a form of deposit in which money is held for a fixed duration of time with the financial institution. A client will subscribe to a term deposit or not is dependent on a large number of features of a client. Banks generally have this information of clients which can help to predict whether a client will subscribe to a term deposit or not. This is an interesting problem which can be solved by analysing the data and building model to predict such clients behaviour. In this project, we will be analysing a *Bank Marketing* data of a Portuguese banking institution collected from [UCI Machine learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## The Data set

The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The data set contains 4521 instances and has 17 columns. The data set that we use is in our project is located in this [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00222) and can be easily downloaded and extracted using our [python script](https://github.com/UBC-MDS/DMC_Portuguese_Group_402/blob/master/src/get_data.py). The dataset is publicly available for research, the details of which are described in [Moro et al., 2014].

**Data set source** : [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014


## The Research Question

For this project, we will be analysing a predictive research question. Our main research question is:

**Given a client's information, predict whether a client will subscribe for a term deposit or not?**

## Action Plan

This kind of dataset might be not balanced. So, first of all, we will check whether our data is balanced or not. If we have imbalanced data, we should use models that can handle class imbalance data by their nature (like gradient boosting trees) or help other models to handle it (like giving more weights to smaller class or perform oversampling/undersampling).

Secondly, we will look at the features. The data contains 16 features, some of which may not be important. Some features can be removed just by visual inspection or by using some algorithms like Random Forest. For visual inspection on numerical features, we will create a correlation matrix and pariplot with distributions, plot graphs of those features against our target variable during our exploratory data analysis. For the categorical features, we will plot the count of each of the levels for our two target classes to do some manual observations about them.

Thirdly, we will check multicollinearity. For some ML models, multicollinearity is a problem and features that have high correlation with other features may be removed. Even though, in theory, multicollinearity is always bad, in practice, sometimes it doesn't have any bad effect on the model. This should be analyzed and based on it we may or may not remove those features.

We plan to use common classification models such as logistic regression, SVM and Random Forest with proper hyper-parameter tuning as part of the classifiers for this task. We would also like to use LightGBM because of our possible class imbalance problem. We would present our result by reporting accuracy, confusion matrix, AUC score and F1 score.

**EDA File**: This [link](https://github.com/UBC-MDS/DMC_Portuguese_Group_402/blob/master/src/dmc_eda.ipynb) contains file for Exploratory Data analysis.
