# required libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def woe_iv(cat_var, bin_target, input_df):
    lst = []

    for i in range(input_df[cat_var].nunique()):
        val = list(input_df[cat_var].unique())[i]
        lst.append([cat_var, val, input_df[(input_df[cat_var] == val) & (bin_target == 0)].count()[cat_var],
                    input_df[(input_df[cat_var] == val) & (bin_target == 1)].count()[cat_var]])
        woe_df = pd.DataFrame(lst, columns=['cat_variable', 'Value', 'Accepted(Events)', 'Rejected(NonEvents)'])       

    woe_df['Events(%)'] = woe_df['Accepted(Events)'] / (woe_df['Accepted(Events)'].sum())
    woe_df['NonEvents(%)'] = woe_df['Rejected(NonEvents)'] / (woe_df['Rejected(NonEvents)'].sum())
    woe_df['NonEvents(%)'] = woe_df['NonEvents(%)'].apply(lambda x: 1 if x == 0 else x)
    woe_df['Events(%)'] = woe_df['Events(%)'].apply(lambda x: 1 if x == 0 else x)
    woe_df['WoE'] = np.log(woe_df['NonEvents(%)'] / woe_df['Events(%)'])
    woe_df['IV'] = (woe_df['NonEvents(%)'] - woe_df['Events(%)'])*woe_df['WoE']
    return woe_df


# function for woe plots
def woe_plot(woe_df, path, fig_name):
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    _g = sns.barplot(x='Value', y="WoE", data=woe_df)
    _g.set_xticklabels(labels=woe_df['Value'], rotation=45)
    plt.title(fig_name)
    plt.tight_layout()
    # _g.figure.savefig(path + '\\' + fig_name + '.jpeg')
    plt.close()
    # Todo: write a function to save the woe plots


def smart_cluster(woe_df, number_of_groups=3, woe_column_name='WOE', group_name='Groups'):
    
    _data = woe_df.sort_values(by=woe_column_name)

    from sklearn.cluster import KMeans
    # create kmeans object
    kmeans = KMeans(n_clusters=number_of_groups)
    # fit kmeans object to data
    kmeans.fit(np.array(_data[woe_column_name]).reshape(-1, 1))

    _level = [1]
    _count = 1
    a = kmeans.labels_
    for i in range(len(a)-1):
        if a[i] == a[i+1]:
            _level.append(_count)
        else:
            _count += 1
            _level.append(_count)
    _data[group_name] = _level
    return _data
