# author: Gaurav Sinha
# date: 2020-01-22

"""
This script does splitting and preprocessing of the input file and returns
preprocessed test and training data file

Usage: src/preprocessing.py --input_file=<input_file> --out_train_file=<out_train_file> --out_test_file=<out_test_file> [--test_split_size=<test_split_size>] [--unproc_train_req=<unproc_train_req>]

Options:
--input_file=<input_file>                   The location of the input file(with file name) on which preprocessing is to be performed
--out_train_file=<out_train_file>           The location where the preprocessed train file(with file name) is to be stored
--out_test_file=<out_test_file>             The location where the preprocessed test file(with file name) is to be stored
--[test_split_size=<test_split_size>]       The proportion of data requred as test set[default: 0.2]
--[unproc_train_req=<unproc_train_req>]     The flag indicating whether unprocessed train file required or not for EDA[default: True ]
"""

#Importing libraries
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from docopt import docopt

opt = docopt(__doc__)


def main(input_file,out_train_file,out_test_file,test_split_size,unproc_train_req):
    #Initializing variables
    unproc_train_file = None
    
    if test_split_size == None:
        test_split_size=0.2
    else:
        test_split_size = float(test_split_size)
        
    
    if unproc_train_req == None:
        unproc_train_req=True
    else:
        unproc_train_req=bool(unproc_train_req)
        
    #Reading csv file 
    bank_df = pd.read_csv(input_file,sep = ';')
    
    
    bank_train_raw, bank_test_raw  = train_test_split(bank_df, test_size = test_split_size, random_state=42)
    
    
    
    #Removing month,day_of_week and duration attributes 
    categorical_features = [ 'job', 'marital', 'default',  'housing','loan', 'contact',  'poutcome','education','y']
    numeric_features = ['age','balance','duration', 'campaign','pdays','previous']
    
    
    #Transforming categorical variable to one hot encoding and scaling numerical feature
    preprocessor = ColumnTransformer(transformers=[
        ('ss', StandardScaler(), numeric_features),
        ('ohe', OneHotEncoder(drop='first'), categorical_features)])
    
    
    
    bank_train_array = preprocessor.fit_transform(bank_train_raw)
    
    
    
    
    cat_cols = list(preprocessor.named_transformers_['ohe'].get_feature_names(categorical_features))
    cols = numeric_features + cat_cols
    
    
    
    bank_train = pd.DataFrame(data = bank_train_array, columns=cols)
    
    
    
    bank_test_array = preprocessor.transform(bank_test_raw)
    
    
    
    bank_test = pd.DataFrame(data = bank_test_array, columns=cols)
    
    #Writing train and test processed file
    write_file(bank_train,out_train_file)
    write_file(bank_test,out_test_file)
    
    #Writing train unprocessed file for EDA
    if unproc_train_req:
        if unproc_train_file == None:
            split_list = out_train_file.split('.')
            unproc_train_file = split_list[0] + '_unprocessed.csv'
        write_file(bank_train_raw,unproc_train_file)
    
    
    
def split_dir_file(path):
    """
    This function separates file path and file name
    
    Parameters:
    -----------
    path: string
        path of a file with file name
    
    Returns:
    --------
    tuple
        Returns a tuple of directory path and file name
    
    Examples:
    ---------
    >>> split_dir_file('data/clean/banking.csv')
    ('data/clean','banking.csv')
    """
    path_list = path.split('/')
    file_name = path_list[-1]
            
    dir_path = path_list[0]
    for i in path_list[1:-1]:
        dir_path = dir_path+"/"+i
            
    return (dir_path,file_name)
    
    
    
def write_file(df,file_path):
    """
    This function takes in dataframe and write it in specified path
    
    Parameters:
    -----------
    df: pandas.DataFrame
        The dataframe to be written in csv file
    path: string
        Path of a file with file name
    
    Returns:
    --------
    tuple
        Returns a tuple of directory path and file name
    
    Examples:
    ---------
    >>> write_file(df,'data/clean/banking.csv')
    """
    path = split_dir_file(file_path)[0]  
    
    try:
        if os.path.isdir(path) == False:
            os.makedirs(path)
        df.to_csv(file_path,index = False)
    
    except OSError:
        print ("Could not create file ", file_path)
    else:
        print ("Successfully created the file ",file_path)
        
def test_split_dir_file():
    assert split_dir_file('data/clean/banking.csv') == ('data/clean','banking.csv')
    
def test_write_file():
    df = pd.DataFrame({
        'x':[1,2],
        'y':[2,3]
    })
    
    file_path = 'testing/test.csv'
    write_file(df,file_path)
    
    assert os.path.isfile(file_path)==True
    
    os.unlink(file_path)
    os.rmdir('testing')
    

test_split_dir_file()
test_write_file()


        
if __name__ == "__main__":
    main(opt['--input_file'],opt['--out_train_file'],opt['--out_test_file'],opt['--test_split_size'],opt['--unproc_train_req'])

