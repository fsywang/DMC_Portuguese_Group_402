 
# Driver script
# Gaurav Sinha, Jan 2020
#
#This script completes the analysis of 
#portugese financial institution client
# data to predict of a customer will 
#subscribe to a term deposit or not. 
#This script combines data downloading, 
#preprocessing, EDA and analysis script.
# This script requires no input
#
# usage: make all

# run all analysis
all: data/raw/bank.csv data/clean/bank_train.csv data/clean/bank_test.csv data/clean/bank_train_uprocessed.csv reports/count_of_cat_features.png reports/density_plot.png reports/kendall_corr_matrix.png reports/pearson_corr_matrix.png reports/proportion_of_class.png reports/training_report.csv reports/training_report.png

# download script
data/raw/bank.csv : src/get_data.R
	Rscript src/get_data.R --out_dir=data/raw/

# data preprocessing
data/clean/bank_train.csv data/clean/bank_test.csv data/clean/bank_train_uprocessed.csv : data/raw/bank.csv src/preprocessing.py
	python src/preprocessing.py --input_file=data/raw/bank.csv --out_train_file=data/clean/bank_train.csv --out_test_file=data/clean/bank_test.csv
# eda
reports/count_of_cat_features.png reports/density_plot.png reports/kendall_corr_matrix.png reports/pearson_corr_matrix.png reports/proportion_of_class.png : data/clean/bank_train_unprocessed.csv src/eda.py 
	python src/eda.py data/clean/bank_train_unprocessed.csv reports

# ml analysis
reports/training_report.csv reports/training_report.png : data/clean/bank_train.csv data/clean/bank_test.csv src/ml_analysis.py
	python src/ml_analysis.py --output_csv=reports/training_report.csv

# clean up intermediate results file
clean_all : 
	rm -rf -d data
	rm -rf -d reports

clean_data :
	rm -rf -d data




