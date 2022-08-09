import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from datasets import SubclassedDataset
from dataloaders import InfiniteDataLoader
import os

# column name of the id
id_name = 'noduleID'

# column names of the features of interest
numeric_feature_names = ['Area', 'ConvexArea', 'Perimeter', 'ConvexPerimeter', 'EquivDiameter',
                         'MajorAxisLength', 'MinorAxisLength',
                         'Elongation', 'Compactness', 'Eccentricity', 'Solidity', 'Extent',
                         'Circularity', 'RadialDistanceSD', 'SecondMoment', 'Roughness', 'MinIntensity',
                         'MaxIntensity', 'MeanIntensity', 'SDIntensity', 'MinIntensityBG',
                         'MaxIntensityBG', 'MeanIntensityBG', 'SDIntensityBG',
                         'IntensityDifference', 'markov1', 'markov2', 'markov3', 'markov4',
                         'markov5', 'gabormean_0_0', 'gaborSD_0_0', 'gabormean_0_1',
                         'gaborSD_0_1', 'gabormean_0_2', 'gaborSD_0_2', 'gabormean_1_0',
                         'gaborSD_1_0', 'gabormean_1_1', 'gaborSD_1_1', 'gabormean_1_2',
                         'gaborSD_1_2', 'gabormean_2_0', 'gaborSD_2_0', 'gabormean_2_1',
                         'gaborSD_2_1', 'gabormean_2_2', 'gaborSD_2_2', 'gabormean_3_0',
                         'gaborSD_3_0', 'gabormean_3_1', 'gaborSD_3_1', 'gabormean_3_2',
                         'gaborSD_3_2', 'Contrast', 'Correlation', 'Energy', 'Homogeneity',
                         'Entropy', 'x_3rdordermoment', 'Inversevariance', 'Sumaverage',
                         'Variance', 'Clustertendency']

# column name of the malignancy label
label_name = 'malignancy'

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_lidc(data_root='./data/', feature_path='LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv', subclass_path='subclass_labels/subclasses.csv'):
    """
    Loads LIDC data into feature and subclass dataframes
    :param data_root: Directory containing the .csv files
    :param feature_path: Path to the .csv file containing the id, features, and malignancy
    :param subclass_path: Path to the .csv file containing the subclass labels
    :return:
    """
    df = pd.read_csv(data_root + feature_path)
    subclass_df = pd.read_csv(data_root + subclass_path)

    return df, subclass_df


def preprocess_data(df, subclass_df, subclass_column='subclass'):
    """
    Preprocesses the data from load_lidc()
    This includes:
    - Removing all data that is not the id, features, malignancy, and subclass
    - Combining subclass data with the feature data
    - Removing nodules with indeterminate malignancy
    - Reducing other malignancy labels to a binary benign/malignant designation
    - Normalizing the feature data by z-score
    :param df: Dataframe including the id, features, and malignancy
    :param subclass_df: Dataframe containing the subclass data
    :param subclass_column: The column of subclass_df containing the subclass data
    :return: A single dataframe containing the id, normalized features, binary malignancy, and subclass labels
    """
    # select features and labels
    df = df.loc[:, [id_name, *numeric_feature_names, label_name]]

    # remove malignancy = 3 or out of range 1-5
    df = df[df[label_name].isin([1, 2, 4, 5])]

    # remove nodules not in the images for fairness
    df = df[df[id_name].isin(list(map(lambda x: int(x.replace('.txt', '')), os.listdir('data/LIDC(MaxSlices)_Nodules'))))]

    # add subclass data
    df['subclass'] = np.empty(len(df))
    subclass_df.index = subclass_df[id_name].values

    for i in df.index:
        df.at[i, 'subclass'] = subclass_df.at[df.at[i, id_name], subclass_column]

    # binarize the remaining malignancy [1,2] -> 0, [4,5] -> 1
    df[label_name] = [int(m - 3 > 0) for m in df[label_name]]

    # normalize numeric features
    df.loc[:, numeric_feature_names] = StandardScaler().fit_transform(df.loc[:, numeric_feature_names].values)

    return df


def split_to_tensors(df):
    """
    Splits a dataframe into tensors of features, labels, and subclass labels
    :param df: The dataframe to split
    :return: 3 tensors of the data
    """
    # tensorify
    data = torch.FloatTensor(df.loc[:, numeric_feature_names].values).to(device)
    labels = torch.LongTensor(df.loc[:, label_name].values).to(device)
    subclass_labels = torch.LongTensor(df.loc[:, 'subclass'].values).to(device)

    return data, labels, subclass_labels


def create_dataset(data, is_dataframe=True):
    """
    Creates an InfiniteDataLoader from the given data and using the given batch size
    :param data: Data to put in the dataloader, either a dataframe or 3 tensors
    :param batch_size: Batch size to use for the dataloader
    :param is_dataframe: Whether data is a dataframe, if False, the data is assumed to already be in the form (feature tensor, label tensor, subclass tensor)
    :return: A dataloader for the data
    """
    if is_dataframe:
        X, y, c = split_to_tensors(data)
    else:
        X, y, c = data

    return X,y, c

def train_val_test_datasets(df, split_path, batch_size):
    # get train/test flags
    train_split = pd.read_csv(split_path)

    # create train/test dataframes
    train_df = df[df["noduleID"].isin(train_split[train_split["split"] == 0]["noduleID"].values)]
    val_df = df[df["noduleID"].isin(train_split[train_split["split"] == 1]["noduleID"].values)]
    test_df = df[df["noduleID"].isin(train_split[train_split["split"] == 2]["noduleID"].values)]

    train = create_dataset(train_df)
    val = create_dataset(val_df)
    test = create_dataset(test_df)

    return train, val, test
