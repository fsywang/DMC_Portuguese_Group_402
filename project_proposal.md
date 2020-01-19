# Project Proposal

## Introduction

Businesses provide numerous services to their clients. It becomes important for them to know whether their clients need those services or not. Banking is one such sector which provides numerous services to their clients. If a client subscribes to their service, it increases revenue for the bank. One such service offered by banks is *Term Deposit*. A term deposit is a form of deposit in which money is held for a fixed duration of time with the financial institution. A client will subscribe to a term deposit or not is dependent on a large number of features of a client. Banks generally have this information of clients which can help to predict whether a client will subscribe to a term deposit or not. This is an interesting problem which can be solved by analysing the data and building model to predict such clients behaviour. In this project, we will be analysing a *Bank Marketing* data of a Portuguese banking institution collected from [UCI Machine learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

## The Data set

The data is related to direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The data set contains 45211 instances and has 17 features. The data set is in zip format and can be easily downloaded using python script from this [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip). The dataset is publicly available for research, the details of which are described in [Moro et al., 2014].

**Data set source** : [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014



## The Research Question

For this project, we will be analysing a predictive research question. Our main research question is:

**Given a client's information, predict whether a client will subscribe for a term deposit or not?**

## Action Plan

This kind of dataset might be not balanced. So we will check whether our data is balanced or not. The data contains 17 features, out of which all might not be important for us. Some features can be removed just by visual inspection like *Name*. For others we will create a correlation matrix, to check which features are important or not. We will also plot graphs of those features against our target variable during our exploratory data analysis. 

We plan to use logistic regression as one of the classifiers for this task. We would check different classifiers including ensemble ones to improve our accuracy. We would present our result by reporting accuracy, confusion matrix and AUC score. We would also provide a visual representation of our result using an ROC curve of the model.
