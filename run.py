import torch
from loss import ERMLoss, GDROLoss
import models
from utils import data_utils, image_data_utils
from train_eval import run_trials
import pandas as pd
from datetime import datetime
import os
import argparse
from dataloaders import InfiniteDataLoader
from datasets import SubclassedDataset


parser = argparse.ArgumentParser()
parser.add_argument('stratification', help='The stratification type to use, available options are "spic_groups", "malignancy", and "cluster"')
parser.add_argument('--test_name', '-n', default='test', help='The name of the test to run (defaults to "test"), the output files will be saved in the directory ./test_results/[name]/')
parser.add_argument('--e2e', action='store_true', help='Use the image data representation (train CNN "end to end")')
parser.add_argument('--designed', '-d', action='store_true', help='Use the designed feature data representation (optional, if no data representation is specified designed is used by default)')
parser.add_argument('--verbose', '-v', action='store_true', help='Print the progress')
parser.add_argument('--trials', '-t', default=100, help='The number of trials to run, defaults to 100')
parser.add_argument('--set_device', '-s', default=0, help='The sets the gpu to use, defaults to 0')


args = parser.parse_args()

# hyperparameters

lr = 0.0005
wd = 0.005
eta = 0.01
gamma = 1.0



if torch.cuda.is_available():
    torch.cuda.set_device(int(args.set_device))
    device = "cuda"
else:
    device = 'cpu'

now = datetime.now()
results_root_dir = 'test_results/'
test_name = args.test_name
results_dir = results_root_dir + f'{test_name}/'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

optimizer_class = torch.optim.Adam

trials = int(args.trials)
batch_size = 128
split_path = "data/train_test_splits/Nodule_Level_30Splits/nodule_split_all.csv"
subclass_path = 'subclass_labels/subclasses.csv'
subclass_column = args.stratification
feature_path = 'LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv'



verbose = args.verbose

for split_num in range(30):

    if args.e2e:
        model_class = models.TransferModel18
        model_args = [True, False, device]
        epochs = 45

    else:
        model_class = models.NeuralNetwork
        model_args = [64, 36, 2]
        epochs = 100


    if args.e2e:
        train, val, test = image_data_utils.get_features(device=device, split_file=split_path, 
                                                         images=True, subclass=subclass_column, 
                                                         split_num=split_num)

    else:
        df = data_utils.preprocess_data(
            *data_utils.load_lidc(
                data_root='data/',
                feature_path=feature_path,
                subclass_path=subclass_path
            ),
            subclass_column=subclass_column
        )

        train, val, test = data_utils.train_val_test_datasets(df, split_path=split_path, split_num=split_num)




    val_dataloader = InfiniteDataLoader(SubclassedDataset(*val), len(val))
    test_dataloader = InfiniteDataLoader(SubclassedDataset(*test), len(test))

    num_subclasses = len(test_dataloader.dataset.subclasses.unique())
    subtypes = ["Overall"]
    if subclass_column == 'cluster':
        subtypes.extend(["Predominantly Benign", "Somewhat Benign", "Somewhat Malignant", "Predominantly Malignant"])
    elif subclass_column == 'spic_groups':
        subtypes.extend(["Unspiculated benign", "Spiculated benign", "Spiculated malignant", "Unspiculated malignant"])
    elif subclass_column == 'malignancy':
        subtypes.extend(["Highly Unlikely", "Moderately Unlikely", "Moderately Suspicious", "Highly Suspicious"])
    else:
        subtypes.extend(list(range(num_subclasses)))

    erm_class = ERMLoss
    erm_args = [None, torch.nn.CrossEntropyLoss()]
    gdro_class = GDROLoss
    gdro_args = [None, torch.nn.CrossEntropyLoss(), eta, num_subclasses]

    optimizer_args = {'lr': lr, 'weight_decay': wd}

    results = {"Accuracies": {}, "Best_Test_Accuracy": {}, "Best_Epochs":{}}

    for loss_class, loss_args in zip([erm_class, gdro_class], [erm_args, gdro_args]):

        fn_name = loss_class.__name__

        tr = SubclassedDataset(*train)

        if fn_name == erm_class.__name__:
            train_dataloader = InfiniteDataLoader(tr, batch_size)
        else:
            train_dataloader = InfiniteDataLoader(tr, batch_size, weights=image_data_utils.get_sampler_weights(tr.subclasses))


        if verbose:
            print(f"Running trials: {fn_name}")

        accuracies, accuracies_best, best_epochs = run_trials(
            num_trials=trials,
            epochs=epochs,
            train_dataloader=train_dataloader,
            val_dataloader = val_dataloader,
            test_dataloader=test_dataloader,
            model_class=model_class,
            model_args=model_args,
            loss_class=loss_class,
            loss_args=loss_args,
            optimizer_class=optimizer_class,
            optimizer_args=optimizer_args,
            device=device,
            verbose=verbose,
            record=True,
            num_subclasses=num_subclasses
        )
        results["Accuracies"][fn_name] = accuracies
        results["Best_Test_Accuracy"][fn_name] = accuracies_best
        results["Best_Epochs"][fn_name] = best_epochs


    accuracies_df = pd.DataFrame(
        results["Accuracies"],
        index=pd.MultiIndex.from_product(
            [range(trials), range(epochs), subtypes],
            names=["trial", "epoch", "subtype"]
        )
    )

    accuracies_BEST_df = pd.DataFrame(
        results["Best_Test_Accuracy"],
        index=pd.MultiIndex.from_product(
        [range(trials),subtypes],
        names=["trial", "subtype"]
        )   
    )

    best_epochs_df = pd.DataFrame(
        results["Best_Epochs"],
        index=pd.MultiIndex.from_product(
        [range(trials)],
        names=["trial"]
        )  
    )
    



    accuracies_df.to_csv(results_dir + f'accuracies_split_{split_num}.csv')
    accuracies_BEST_df.to_csv(results_dir + f'accuracies_BEST_split_{split_num}.csv')
    best_epochs_df.to_csv(results_dir + f'best_epochs_split_{split_num}.csv')
