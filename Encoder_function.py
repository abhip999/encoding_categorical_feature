import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Encoding:
    """
    To handle conversion of categorical variables to numerical values
    """
    def __init__(self,  df):
        """
        Constructor of the class
        :param df: Dataframe for encoding
        """
        self.df = df

    def label_encoding(self, column_name):
        """
        Label encoding of the ordinal feature of the dataframe
        :param column_name: Column name that need to be encoded as label encoder
        :return: Dataframe column with label encoded feature
        """
        le = LabelEncoder()
        le.fit(self.df[column_name].drop_duplicates())
        self.df[column_name] = le.transform(self.df[column_name])
        return self.df

    def create_dummy_variable(self, column_name, inplace1=True) -> pd.DataFrame:
        """
        Creation of dummy variables for nominal feature of the dataframe
        :param column_name: Column name for which dummy variable need to be created
        :param inplace1: True or False
        :return: (k-1) Dummy varibles for feature having k unique values
        """
        dummy_values = pd.get_dummies(self.df[column_name], drop_first=True, prefix=column_name)
        dataframe = pd.concat([self.df, dummy_values], axis=1)
        dataframe.drop([column_name], axis=1, inplace=inplace1)

        return dataframe

    def inplace_category(self, column_name, unique_value_list, change_to_list):
        """
        For Changing column name of dummy variable created
        :param column_name: Feature name
        :param unique_value_list: Unique names in feature
        :param change_to_list: List for changing column name
        :return: Dataframe
        """
        self.df[column_name].replace(to_replace=unique_value_list, value=change_to_list, inplace=True)
        return self.df
