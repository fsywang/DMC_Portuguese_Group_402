# Marketing campaigns Subscription 
Shiying Wang,  Karlos Muradyan, Gaurav Sinha

# Summary 

In this project data used is related to direct marketing campaigns of a Portuguese banking institution.These marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The data set contains 4521 instances and has 17 columns. The data set that we use is in our project is located in this [link](https://archive.ics.uci.edu/ml/machine-learning-databases/00222) and can be easily downloaded and extracted using our [python script](https://github.com/UBC-MDS/DMC_Portuguese_Group_402/blob/master/src/get_data.py). The dataset is publicly available for research, the details of which are described in [Moro et al., 2014].

# Report

The final report can be found [here](https://github.com/UBC-MDS/DMC_Portuguese_Group_402/blob/master/doc/report.ipynb).

# Usage

To replicate the analysis, clone this GitHub repository, install all [dependencies](https://github.com/UBC-MDS/DMC_Portuguese_Group_402/blob/master/requirements.txt), and run the following commands:

```
# download data
Rscript src/get_data.R

# data preprocessing
python src/preprocessing.py --input_file=data/raw/bank.csv --out_train_file=data/clean/bank_train.csv --out_test_file=data/bank_test.csv


# eda
python src/eda.py "data/clean/bank_train_unprocessed.csv" "reports"

# predictive models


```

# Dependencies

The dependencies can be found [here](https://github.com/UBC-MDS/DMC_Portuguese_Group_402/blob/master/requirements.txt)

# Reference

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

UCI Machine Learning Repository. University of California, Irvine, School of Information; Computer Sciences.https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
