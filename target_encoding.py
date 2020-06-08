import pandas as pd
import numpy as np
from category_encoders import target_encoder as te


def encode_category(category, target, df):
    """
    Target encodes the high cardinality nominal feature
    :param category: High cardinality nominal feature
    :param target: the dependent variable
    :param df: Data frame in which category and target present
    :return: Series of encoded value of the category against the target
    """
    df[category] = df[category].fillna('NoData')
    x = df[category]
    y = df[target]
    ec = te.TargetEncoder()
    x_te = ec.fit_transform(x, y)
    out_df = pd.concat([x, x_te], axis=1)
    out_df.columns = [category, 'value']

    # add some noises to avoid over fitting
    control = 0.3
    capped = df[target].mean() * control
    num_obs = df.shape[0]
    np.random.seed(0)
    noise = np.random.uniform(0, capped, num_obs)
    out_df['value'] = out_df['value'] + noise
    return out_df
