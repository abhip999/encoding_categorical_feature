# required libraries

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


# function for calculating weight of evidence
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

    woe_df['IV'] = (woe_df['NonEvents(%)'] - woe_df['Events(%)']) * woe_df['WoE']
    return woe_df


# Function for smart cluster on multiclass
def smart_cluster_multiclass(data, woe_column_name='WOE', number_of_cluster=2, group_name='Groups'):
    _data = data.sort_values(by=woe_column_name)
    from sklearn.cluster import KMeans
    _x = np.array(_data['WoE']).reshape(-1, 1)

    kmeans = KMeans(n_clusters=number_of_cluster)
    # fit kmeans object to data
    kmeans.fit(np.array(_x).reshape(-1, 1))

    _level = [1]
    _count = 1
    a = kmeans.labels_
    for i in range(len(a) - 1):
        if a[i] == a[i + 1]:
            _level.append(_count)
        else:
            _count += 1
            _level.append(_count)
    _data[group_name] = _level
    return _data


# Function for woe multiclas
def woe_multiclass(df, nominal_column, target, path, mapping_cluster_list):
    _data = df[[nominal_column, target]]
    nominal_class = df[target].unique()
    count = 1
    for i in nominal_class:
        _data['nominal_column + '_class' + str(i)] = \
            _data[target].apply(lambda x: 1 if x == i else 0)
    target = list(_data.columns)[2:]

    for i in range(len(target)):
        df_nominal = woe_iv(nominal_column, _data[target[i]], _data)

        # plt.figure(figsize=(10, 18))
        # plt.subplot(n, 1, i + 1)
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})
        g = sns.barplot(x='Value', y="WoE", data=df_nominal)
        g.set_xticklabels(labels=df_nominal['Value'], rotation=45)
        plt.title(target[i])
        plt.tight_layout()
        # g.figure.savefig(path + '\\' + target[i] + '.jpeg')
        plt.close()

        groups_name = target[i] + '_Groups'
        df_nominal = smart_cluster_multiclass(df_nominal, 'WoE', mapping_cluster_list[i], group_name=groups_name)

        if count == 1:
            _df = df_nominal[['Value', groups_name]]
        else:
            _df[groups_name] = df_nominal[groups_name]
        count += 1
    return _df
