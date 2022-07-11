import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from datasets import SubclassedNoduleDataset
from dataloaders import InfiniteDataLoader, SubtypedDataLoader

# numeric_data_by_radiologist_path = 'data/LIDC_20130817_AllFeatures2D_MaxSlice_MattEdited.csv'
numeric_data_path = 'data/LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv'
# max_slice_data_path = 'data/LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv'
subtype_data_path = 'data/LIDC_spic_subgrouped.csv'

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
subclass_label_name = 'subgroup'
subclasses = ['unmarked_benign', 'marked_benign', 'marked_malignant', 'unmarked_malignant']
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    df = pd.read_csv(numeric_data_path)
    # max_slice_df = pd.read_csv(max_slice_data_path)
    # max_slice_df.index = max_slice_df[id_name]

    subtype_df = pd.read_csv(subtype_data_path)

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
    subclass_labels = torch.LongTensor(
        list(map(lambda x: subclasses.index(x), df.loc[:, subclass_label_name].values))).to(device)

    return data, labels, subclass_labels


def create_dataloader(df, batch_size):
    data, labels, subclass_labels = split_to_tensors(df)

    # wrap with dataset and dataloader
    dataloader = InfiniteDataLoader(SubclassedNoduleDataset(data, labels, subclass_labels), batch_size=batch_size)

    return dataloader


def create_subtyped_dataloader(df, subtype_df, batch_size, proportional):
    def get_subtype_data(subtype_name):
        # get all rows of df where the nodule id is associated with subtype_name
        return df.loc[
               [subtype_df.at[nodule_id, "subgroup"] == subtype_name for nodule_id in df[id_name]], :]

    # df = df[df[id_name].isin(subtype_df[id_name])]
    # subtype_df = subtype_df[subtype_df[id_name].isin(df[id_name])]

    subtype_names = ["unmarked_benign", "marked_benign", "marked_malignant", "unmarked_malignant"]
    subtype_dfs = [get_subtype_data(name) for name in subtype_names]

    # separate into training and test sets
    subtype_data = []
    for subtype in subtype_dfs:
        data, labels, _ = split_to_tensors(subtype)
        subtype_data.append((data, labels))

    # wrap with dataset and dataloader
    dataloader = SubtypedDataLoader(subtype_data, batch_size, total=proportional)

    return dataloader
