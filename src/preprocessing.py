# author: Gaurav Sinha
# date: 2020-01-22

"""
This script does splitting and preprocessing of the input file and returns
preprocessed test and training data file

Usage: preprocessing.py <arg1> --arg2=<arg2> [--arg3=<arg3>]

Options:
<arg>             Takes any value (this is a required positional argument)
--arg2=<arg2>     Takes any value (this is a required option)
[--arg3=<arg3>]   Takes any value (this is an optional option)

"""


import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


def main():
    input_file = '../data/raw/bank.csv'
    out_train_file = 'data/clean/bank_train.csv'
    out_test_file = 'data/clean/bank_test.csv'
    test_split_size = 0.2
    unproc_train_req = True
    unproc_test_req = False
    unproc_train_file = None
    unproc_test_file = None
    
    
    
    bank_df = pd.read_csv(input_file,sep = ';')
    
    
    bank_train_raw, bank_test_raw  = train_test_split(bank_df, test_size = test_split_size, random_state=42)
    
    
    
    #Removing month,day_of_week and duration attributes 
    categorical_features = [ 'job', 'marital', 'default',  'housing','loan', 'contact',  'poutcome','education','y']
    numeric_features = ['age','balance','duration', 'campaign','pdays','previous']
    
    
    
    preprocessor = ColumnTransformer(transformers=[
        ('ss', StandardScaler(), numeric_features),
        ('ohe', OneHotEncoder(drop='first'), categorical_features)])
    
    
    
    bank_train_array = preprocessor.fit_transform(bank_train_raw)
    
    
    
    
    cat_cols = list(preprocessor.named_transformers_['ohe'].get_feature_names(categorical_features))
    cols = numeric_features + cat_cols
    
    
    
    bank_train = pd.DataFrame(data = bank_train_array, columns=cols)
    
    
    
    bank_test_array = preprocessor.transform(bank_test_raw)
    
    
    
    bank_test = pd.DataFrame(data = bank_test_array, columns=cols)
    
    
    write_file(bank_train,out_train_file)
    write_file(bank_test,out_test_file)
    
    if unproc_train_req:
        if unproc_train_file == None:
            split_list = out_train_file.split('.')
            unproc_train_file = split_list[0] + '_unprocessed.csv'
        write_file(bank_train_raw,unproc_train_file)
    
    if unproc_test_req :
        if unproc_test_file == None:
            split_list = out_test_file.split('.')
            unproc_test_file = split_list[0] + '_unprocessed.csv'
        write_file(bank_test_raw,unproc_test_file)
    
    
def split_dir_file(path):
    path_list = path.split('/')
    file_name = path_list[-1]
            
    dir_path = path_list[0]
    for i in path_list[1:-1]:
        dir_path = dir_path+"/"+i
            
    return (dir_path,file_name)
    
    
    
def write_file(df,file_path):
        
    path = split_dir_file(file_path)[0]  
    
    try:
        if os.path.isdir(path) == False:
            os.makedirs(path)
        df.to_csv(file_path,index = False)
    
    except OSError:
        print ("Could not create file ", file_path)
    else:
        print ("Successfully created the file ",file_path)
        
if __name__ == "__main__":
    main()
