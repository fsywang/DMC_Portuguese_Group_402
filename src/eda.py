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
import numpy as np


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
        y = alt.Y('y:O',title = '')).configure_title(fontSize=20).configure_axis(labelFontSize=13,titleFontSize=15)
    
    p0.save( out_dir + "/proportion_of_class.png")
    
    #pearson corrlation matrix plot
    sns.set(rc={'figure.figsize':(8,6)},font_scale=1.3)
    pearson_corr_matrix = sns.heatmap(bank_train.corr().round(2), annot=True,
           mask = np.triu(np.ones_like(bank_train.corr(), dtype=np.bool), 1),
            cmap = sns.diverging_palette(200, 16, as_cmap=True)
    ).set_title('Pearson correlation matrix',fontsize = 25) 
    p1 = pearson_corr_matrix.get_figure()
    p1.savefig( out_dir + "/pearson_corr_matrix.png")
    
    plt.clf()
    #kendall corrlation matrix plot
    sns.set(rc={'figure.figsize':(8,6)},font_scale=1.3)
    kendall_corr_matrix = sns.heatmap(bank_train.corr(method='kendall').round(2), annot=True,
           mask = np.triu(np.ones_like(bank_train.corr(method='kendall'), dtype=np.bool), 1),
            cmap = sns.diverging_palette(200, 16, as_cmap=True)
    ).set_title('Kendall correlation matrix',fontsize = 25)
    p2 = kendall_corr_matrix.get_figure()
    p2.savefig( out_dir + "/kendall_corr_matrix.png")


    #density plot
    density_plot= (make_num_plot(bank_train,'pdays') | make_num_plot(bank_train,'duration')).configure_title(fontSize=20).configure_axis(
    labelFontSize=13,titleFontSize=15)
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
        alt.X("count()"),
        alt.Y(alt.repeat("row"), type = 'nominal'),
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