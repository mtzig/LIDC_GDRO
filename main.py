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
test_csv = "MaxSliceTestSetPreprocessed.csv"
train_csv = "MaxSliceTrainingValidationSetPreprocessed.csv"

device = "cuda" if torch.cuda.is_available() else "cpu"
training_fraction = 0.8
batch_size = 40
proportional = True

is_gdro = True

groupdro_hparams = {"groupdro_eta": 0.1}

# if true, will randomly split test and training/validation data and save to csv
# changing the feature names will require reshuffling the data to update the csvs
shuffle_data = False


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
        return df.loc[
               [subtype_df.at[nodule_id, "subtype"] == subtype_name
                if nodule_id in subtype_df["Nodule_id"].values else False
                for nodule_id in df[id_name]], :]

    subtype_names = ["0benign", "1benign", "0malignant", "1malignant"]
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
    subtype_df = pd.read_csv("lidc_subtyped.csv")

    if shuffle_data:
        # import data
        df = pd.read_csv("LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv")

        # preprocess data
        df = preprocess_data(df)

        training_df = df.sample(frac=training_fraction)
        test_df = df.drop(training_df.index)
        training_df.to_csv("MaxSliceTrainingValidationSetPreprocessed.csv")
        test_df.to_csv("MaxSliceTestSetPreprocessed.csv")
    else:
        training_df = pd.read_csv("MaxSliceTrainingValidationSetPreprocessed.csv")
        test_df = pd.read_csv("MaxSliceTestSetPreprocessed.csv")

    subtype_df.index = subtype_df["Nodule_id"].values

    N = 120

    results = [[], []]
    for is_gdro in [0, 1]:

        print("Running test: " + ["ERM", "GDRO"][is_gdro])

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

            if is_gdro:
                loss_fn = loss.GDROLoss(model, torch.nn.CrossEntropyLoss(), groupdro_hparams)
            else:
                loss_fn = loss.ERMLoss(model, torch.nn.CrossEntropyLoss(), {})
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)

            epochs = 40

            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")
                train.train(train_dataloader, model, loss_fn, optimizer)
                # train.test(test_dataloader, model)

            results[is_gdro].append(train.test(test_dataloader, model))

    results_df = pd.DataFrame(results)
    results_df.to_csv("results")

    print("Test complete")



if __name__ == "__main__":
    main()
