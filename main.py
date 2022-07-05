from torch.utils.data import DataLoader

import loss
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import models
import train
from datasets import NoduleDataset
from dataloaders import InfiniteDataLoader, SubtypedDataLoader

id_name = 'noduleID'
feature_names = ['Area', 'ConvexArea', 'Perimeter', 'ConvexPerimeter', 'EquivDiameter',
                 'MajorAxisLength', 'MinorAxisLength', 'SuperscribedDiameter',
                 'Elongation', 'Compactness', 'Eccentricity', 'Solidity', 'Extent',
                 'Circularity', 'RadialDistanceSD', 'SecondMoment', 'Roughness',
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
label_name = 'malignancy'
all_data_csv = "LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

training_fraction = 0.8
batch_size = 160
proportional = True

is_gdro = True

hparams = {"groupdro_eta": 0.1}


def preprocess_data(df):
    # select features and labels
    df = df.loc[:, [id_name, *feature_names, label_name]]

    # remove malignancy = 3 or out of range 1-5
    df = df[df[label_name].isin([1, 2, 4, 5])]

    # binarize the remaining malignancy [1,2] -> 0, [4,5] -> 1
    df[label_name] = [int(m - 3 > 0) for m in df[label_name]]

    # normalize numeric features
    df.loc[:, feature_names] = StandardScaler().fit_transform(df.loc[:, feature_names].values)

    return df


def split_to_tensors(df):
    # tensorify
    data = torch.FloatTensor(df.loc[:, feature_names].values).to(device)
    labels = torch.LongTensor(df.loc[:, label_name].values).to(device)

    return data, labels


def create_dataloader(df):
    data, labels = split_to_tensors(df)

    # wrap with dataset and dataloader
    dataloader = InfiniteDataLoader(NoduleDataset(data, labels), batch_size=batch_size)

    return dataloader


def create_subtyped_dataloader(df, subtype_df):
    def get_subtype_data(subtype_name):
        # get all rows of df where the nodule id is associated with subtype_name
        return df.loc[
               [subtype_df.at[nodule_id, "subgroup"] == subtype_name for nodule_id in df[id_name]], :]

    #df = df[df[id_name].isin(subtype_df[id_name])]
    #subtype_df = subtype_df[subtype_df[id_name].isin(df[id_name])]

    subtype_names = ["unmarked_benign", "marked_benign", "marked_malignant", "unmarked_malignant"]
    subtype_dfs = [get_subtype_data(name) for name in subtype_names]

    # separate into training and test sets
    subtype_data = []
    for subtype in subtype_dfs:
        data, labels = split_to_tensors(subtype)
        subtype_data.append((data, labels))

    # wrap with dataset and dataloader
    dataloader = SubtypedDataLoader(subtype_data, batch_size, total=proportional)

    return dataloader


def main():
    subtype_df = pd.read_csv("data/lidc_spic_subgrouped_radiologist.csv")

    # import data
    df = pd.read_csv("data/LIDC_individual_radiologists.csv")
    # preprocess data (normalization, remove anything that isn't in the chosen features)
    df = preprocess_data(df)

    # import train/test flags
    train_test = pd.read_csv("data/lidc_train_test_radiologist.csv")

    # create train/test dataframes
    training_df = df[df["noduleID"].isin(train_test[train_test["dataset"] == "train"]["noduleID"].values)]
    test_df = df[df["noduleID"].isin(train_test[train_test["dataset"] == "test"]["noduleID"].values)]

    # use noduleIDs as index, it makes things easier
    subtype_df.index = subtype_df["noduleID"].values

    N = 60

    results = [[], [], []]
    for is_gdro in []:#[0, 1, 2]:

        print("Running test: " + ["ERM", "GDRO", "Combined"][is_gdro])

        # create the training and testing dataloaders
        if is_gdro:
            train_dataloader = create_subtyped_dataloader(training_df, subtype_df)
        else:
            train_dataloader = create_dataloader(training_df)

        test_dataloader = create_subtyped_dataloader(test_df, subtype_df)

        for i in range(N):

            print(f"Trial {i + 1}/{N}")

            # create and train model
            model = models.NeuralNetwork(64, 32, 32, 2)
            model.to(device)

            if is_gdro == 0:
                loss_fn = loss.ERMLoss(model, torch.nn.CrossEntropyLoss(), hparams)
            elif is_gdro == 1:
                loss_fn = loss.GDROLoss(model, torch.nn.CrossEntropyLoss(), hparams)
            else:
                loss_fn = loss.ERMGDROLoss(model, torch.nn.CrossEntropyLoss(), hparams)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)

            epochs = 40

            for epoch in range(epochs):
                # print(f"Epoch {epoch + 1}/{epochs}")
                train.train(train_dataloader, model, loss_fn, optimizer)
                # train.test(test_dataloader, model)

            results[is_gdro].append(train.test(test_dataloader, model))

    results_df = pd.DataFrame(results)
    results_df.to_csv("results.csv")

    print("Test complete")


if __name__ == "__main__":
    main()
