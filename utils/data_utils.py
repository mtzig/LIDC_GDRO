import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from datasets import SubclassedDataset
from dataloaders import InfiniteDataLoader

numeric_data_path = 'LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv'
subclass_data_path = 'subclass_labels/LIDC_data_split_with_cluster.csv'

id_name = 'noduleID'
radiologist_id_name = 'RadiologistID'
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
# numeric_feature_names += [f'CNN_{n}' for n in range(1, 37)]
semantic_feature_names = ['subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']
label_name = 'malignancy'
subclass_label_name = 'subclass'
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_lidc(data_root):
    df = pd.read_csv(data_root + numeric_data_path)
    # max_slice_df = pd.read_csv(max_slice_data_path)
    # max_slice_df.index = max_slice_df[id_name]

    subtype_df = pd.read_csv(data_root + subclass_data_path)

    # attach malignancy features to the numeric feature dataframe
    # for instance in df.index:
    #     nodule_id = df.at[instance, id_name]
    #     radiologist = df.at[instance, radiologist_id_name]
    #     df.at[instance, label_name] = max_slice_df.at[nodule_id, label_name + f'_{radiologist}']

    return df, subtype_df


def preprocess_data(df, subtype_df):
    # select features and labels
    df = df.loc[:, [id_name, *numeric_feature_names, label_name]]

    # add subclass data
    df[subclass_label_name] = np.empty(len(df))
    subtype_df.index = subtype_df[id_name].values

    df = df[df[id_name].isin(subtype_df[id_name])]

    for i in df.index:
        df.at[i, subclass_label_name] = subtype_df.at[df.at[i, id_name], subclass_label_name]

    # remove malignancy = 3 or out of range 1-5
    df = df[df[label_name].isin([1, 2, 4, 5])]

    # binarize the remaining malignancy [1,2] -> 0, [4,5] -> 1
    df[label_name] = [int(m - 3 > 0) for m in df[label_name]]

    # normalize numeric features
    df.loc[:, numeric_feature_names] = StandardScaler().fit_transform(df.loc[:, numeric_feature_names].values)

    return df


def split_to_tensors(df):
    # tensorify
    data = torch.FloatTensor(df.loc[:, numeric_feature_names].values).to(device)
    labels = torch.LongTensor(df.loc[:, label_name].values).to(device)
    subclass_labels = torch.LongTensor(df.loc[:, subclass_label_name].values).to(device)

    return data, labels, subclass_labels


def create_dataloader(data, batch_size, is_dataframe=True):
    if is_dataframe:
        X, y, c = split_to_tensors(data)
    else:
        X, y, c = data

    # wrap with dataset and dataloader
    dataloader = InfiniteDataLoader(SubclassedDataset(X, y, c), batch_size=batch_size)

    return dataloader