import loss
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import models
import train
from datasets import NoduleDataset, SubtypedDataLoader
from fast_data_loader import InfiniteDataLoader

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
device = "cuda" if torch.cuda.is_available() else "cpu"
training_fraction = 0.8
batch_size = 4
epoch_size = 23

is_gdro = False

groupdro_hparams = {"groupdro_eta": 0}


def preprocess_data(df):
    # select features and labels
    df = df.loc[:, [id_name, *feature_names, label_name]]

    # remove malignancy = 3
    df = df[df[label_name] != 3]

    # binarize the remaining malignancy [1,2] -> 0, [4,5] -> 1
    df[label_name] = [int(m - 3 > 0) for m in df[label_name]]

    # normalize numeric features
    df.loc[:, feature_names] = StandardScaler().fit_transform(df.loc[:, feature_names].values)

    return df


def split_to_tensors(df, frac):
    # separate into training and test sets
    training_df = df.sample(frac=frac)
    test_df = df.drop(training_df.index)

    # tensorify
    training_data = torch.FloatTensor(training_df.loc[:, feature_names].values).to(device)
    training_labels = torch.LongTensor(training_df.loc[:, label_name].values).to(device)
    test_data = torch.FloatTensor(test_df.loc[:, feature_names].values).to(device)
    test_labels = torch.LongTensor(test_df.loc[:, label_name].values).to(device)

    return training_data, training_labels, test_data, test_labels


def create_dataloaders(df):
    training_data, training_labels, test_data, test_labels = split_to_tensors(df, training_fraction)

    # wrap with datasets and dataloaders
    train_dataloader = iter(InfiniteDataLoader(NoduleDataset(training_data, training_labels), batch_size=batch_size))
    test_dataloader = iter(InfiniteDataLoader(NoduleDataset(test_data, test_labels), batch_size=batch_size))

    return train_dataloader, test_dataloader


def create_subtyped_dataloaders(df, subtype_df):
    def get_subtype_data(subtype_name):
        return df.loc[
               [subtype_df.at[nodule_id, "subtype"] == subtype_name
                if nodule_id in subtype_df["Nodule_id"].values else False
                for nodule_id in df[id_name]], :]

    subtype_names = subtype_df["subtype"].unique()
    subtype_dfs = {name: get_subtype_data(name) for name in subtype_names}

    # separate into training and test sets
    training_subtype_data = test_subtype_data = {}
    for name in subtype_dfs:
        training_data, training_labels, test_data, test_labels = split_to_tensors(subtype_dfs[name], training_fraction)

        training_subtype_data[name] = (training_data, training_labels)
        test_subtype_data[name] = (test_data, test_labels)

    # wrap with datasets and dataloaders
    train_dataloader = SubtypedDataLoader(training_subtype_data, batch_size=batch_size)
    test_dataloader = SubtypedDataLoader(test_subtype_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def main():
    # import data
    df = pd.read_csv("LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv")
    subtype_df = pd.read_csv("lidc_subtyped.csv")

    # preprocess data
    df = preprocess_data(df)
    subtype_df.index = subtype_df["Nodule_id"].values

    # create the training and testing dataloaders
    if is_gdro:
        train_dataloader, test_dataloader = create_subtyped_dataloaders(df, subtype_df)
    else:
        train_dataloader, test_dataloader = create_dataloaders(df)

    # create and train model
    model = models.NeuralNetwork(64, 32, 32, 2)

    if is_gdro:
        loss_fn = loss.GDROLoss(model, torch.nn.CrossEntropyLoss(), groupdro_hparams)
    else:
        loss_fn = loss.ERMLoss(model, torch.nn.CrossEntropyLoss(), {})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    epochs = 100

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train.train(train_dataloader, epoch_size, model, loss_fn, optimizer)
        train.test(test_dataloader, epoch_size, batch_size, model, loss_fn, is_gdro)
    print("Done!")


if __name__ == "__main__":
    main()
