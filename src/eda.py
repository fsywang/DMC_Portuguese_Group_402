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
    #pariplot
    pairplot_numeric = sns.pairplot(bank_train,hue='y')
    pairplot_numeric.savefig( out_dir + "/pairplot_numeric.png")
    # plot count of categorical features
    def make_cat_plot(cat_list):
        """
        plot count of categorical features.

        Parameters
        ----------
            list: cat_list 
            list of strings contains features name

        Returns
        -------
        altair.vegalite.v3.api.Chart
            altair plots

        Examples
        --------
            make_cat_plot(['job', 'marital', 'education'])

        """

        cat_p  = alt.Chart(bank_train).mark_bar(opacity = 0.8).encode(
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
    p = make_cat_plot(['job', 'marital', 'education']) | make_cat_plot(['default', 'housing', 'loan']) |make_cat_plot(['contact', 'poutcome', 'month'])
    p

    p.save( out_dir + "/count_of_cat_features.png")
    

# call main function
if __name__ == "__main__":
    main(opt["<train>"], opt["<out_dir>"])