# author: Shiying Wang
# date: 2020-01-22

'''This script produce explotary data analysis and returns plots. 
This script takes a filename as the arguments.

Usage: eda.py <train> <out_dir>
  
Options:
<train> Path (including filename) to training data
<out_dir> Path to directory where the plots should be saved
'''

import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import os
from docopt import docopt


opt = docopt(__doc__)
# define main function
def main(train, out_dir):
    #test input 
    assert out_dir.endswith('/') == False, "out_dir cannot end with '/'"
    
    #load data
    bank_train = pd.read_csv(train)

    # tests case
    assert bank_train.shape[1] == 17, "should have 17 columns"
    assert all(bank_train.columns == ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'y']), "input column name is wrong'"

    # check if the folder exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    #proportion plot
    p0 = alt.Chart(bank_train, title = "Proportion of two classes").mark_bar().encode(
        x = alt.X('count()'),
        y = alt.Y('y:O'))
    
    p0.save( out_dir + "/proportion_of_class.png")
    
    #pearson corrlation matrix plot
    pearson_corr_matrix = sns.heatmap(bank_train.corr(), annot=True)
    pearson_corr_matrix.set_title('Pearson correlation matrix')
    p1 = pearson_corr_matrix.get_figure()
    p1.savefig( out_dir + "/pearson_corr_matrix.png")
    
    plt.clf()
    #kendall corrlation matrix plot
    kendall_corr_matrix = sns.heatmap(bank_train.corr(method='kendall'), 
                                      annot=True)
    kendall_corr_matrix.set_title('Kendall correlation matrix')
    p2 = kendall_corr_matrix.get_figure()
    p2.savefig( out_dir + "/kendall_corr_matrix.png")


    #density plot
    density_plot= make_num_plot(bank_train, 'pdays') | make_num_plot(bank_train, 'duration')
    density_plot.save( out_dir + "/density_plot.png")
    # plot count
    p = make_cat_plot(bank_train, ['job', 'default']) | make_cat_plot(bank_train, ['education', 'loan']) | make_cat_plot(bank_train, ['marital', 'housing'])
    p

    p.save( out_dir + "/count_of_cat_features.png")
    
 # plot count of categorical features
def make_cat_plot(dat, cat_list):
    """
    plot count of categorical features.

    Parameters
    ----------
    list: cat_list 
        list of strings contains features name
    DataFrame: dat
        input dataset

    Returns
    -------
    altair.vegalite.v3.api.Chart
        altair plots

    Examples
    --------
        make_cat_plot(bank_train, ['job', 'default'])

    """

    cat_p  = alt.Chart(dat).mark_bar(opacity = 0.8).encode(
        alt.X(alt.repeat("row"), type = 'nominal'),
        alt.Y("count()"),
        color='y'
    ).properties(
            width=200,
            height=150
        ).repeat(
        row = cat_list
    )     
    return cat_p   

def make_num_plot(dat, col):
    """
    density plot of numerical features.

    Parameters
    ----------
    string: col 
        column names
    DataFrame: dat
        input dataset

    Returns
    -------
    altair.vegalite.v3.api.Chart
        altair plots

    Examples
    --------
        make_num_plot(bank_train, 'duration')

    """
    p = alt.Chart(dat, title = "Density plot of {}".format(col)).transform_density(
    col,
    as_=[col, 'density'],
    groupby=['y']
    ).mark_area(fillOpacity=0.3).encode(
        x=col,
        y='density:Q',
        color='y'
    ).properties(
            width=400,
            height=400
        )
    
    return p
    
# call main function
if __name__ == "__main__":
    main(opt["<train>"], opt["<out_dir>"])