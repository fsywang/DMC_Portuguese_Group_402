# author: Shiying Wang
# date: 2020-01-22

'''This script produce explotary data analysis and returns plots. 
This script takes a filename as the arguments.

Usage: eda.py <train> <out_dir>
  
Options:
<train>     Path (including filename) to training data
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
    #load data
    bank_train = pd.read_csv(train, sep = ";")
    # check if the folder exist
    if not os.path.exists("../" + out_dir):
        os.mkdir("../" + out_dir)
    #proportion plot
    p0 = alt.Chart(bank_train, title = "Proportion of two classes").mark_bar().encode(
        x = alt.X('count()'),
        y = alt.Y('y:O'))
    
    p0.save("../"+ out_dir + "/proportion_of_class.png")
    
    #pearson corrlation matrix plot
    pearson_corr_matrix = sns.heatmap(bank_train.corr(), annot=True)
    pearson_corr_matrix.set_title('Pearson correlation matrix')
    p1 = pearson_corr_matrix.get_figure()
    p1.savefig("../"+ out_dir + "/pearson_corr_matrix.png")
    
    plt.clf()
    #kendall corrlation matrix plot
    kendall_corr_matrix = sns.heatmap(bank_train.corr(method='kendall'), 
                                      annot=True)
    kendall_corr_matrix.set_title('Kendall correlation matrix')
    p2 = kendall_corr_matrix.get_figure()
    p2.savefig("../"+ out_dir + "/kendall_corr_matrix.png")
    #pariplot
    pairplot_numeric = sns.pairplot(bank_train,hue='y')
    pairplot_numeric.savefig("../"+ out_dir + "/pairplot_numeric.png")
    #categorical features columns
    cat_columns = ['job', 'marital', 'education', 'month'
                   'default', 'housing', 'loan', 'contact', 'poutcome']

    for col in cat_columns:
        p = alt.Chart(bank_train, title = "Count of {}".format(col)).mark_bar(opacity = 0.8).encode(
            x = alt.X('count({}):Q'.format(col)),
                y= alt.Y('{}:O'.format(col), 
                  sort=alt.EncodingSortField(
                    field= "{}".format(col),
                    op= "count",
                    order= "descending")),
            color='y')
        p.save("../"+ out_dir + "/count_of_{}.png".format(col))
    
    
    
# call main function
if __name__ == "__main__":
    main(opt["<train>"], opt["<out_dir>"])