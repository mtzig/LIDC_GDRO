import torch
from loss import ERMLoss, GDROLoss
import models
from utils import data_utils, image_data_utils
from train_eval import run_trials_images,run_trials_designedFeatures,run_trials_CNN
import pandas as pd
from datetime import datetime
import os
import argparse
from dataloaders import InfiniteDataLoader
from datasets import SubclassedDataset
import glob
import torch.optim.lr_scheduler as schedulers
import torch.optim as optimizers


parser = argparse.ArgumentParser()
parser.add_argument('stratification', help='The stratification type to use, available options are "spic_groups", "malignancy", and "cluster"')
parser.add_argument('--test_name', '-n', default='test', help='The name of the test to run (defaults to "test"), the output files will be saved in the directory ./test_results/[name]/')
# parser.add_argument('--cnn', action='store_true', help='Use the CNN (deep) feature data representation')
parser.add_argument('--e2e', action='store_true', help='Use the image data representation (train CNN "end to end")')
parser.add_argument('--designed', '-d', action='store_true', help='Use the designed feature data representation (optional, if no data representation is specified designed is used by default)')
parser.add_argument('--verbose', '-v', action='store_true', help='Print the progress')
parser.add_argument('--colab', '-c', action='store_true', help='Add this argument to properly download the results if the script is being run through Google Colab')
parser.add_argument('--trials', '-t', default=30, help='The number of trials to run, defaults to 30')

args = parser.parse_args()


# hyperparameters

lr_erm = 0.0005
lr_dro = 0.0005
wd_erm = 0.005
wd_dro = 0.005
eta = 0.01

# device = 'cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() and torch.backends.mps.is_built() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.e2e:
    model_class = models.TransferModel18
    model_args = [True, False, device]
else:
    model_class = models.NeuralNetwork
    model_args = [64, 36, 2]

optimizer_class = torch.optim.Adam

trials = int(args.trials)

epochs = 45
batch_size = 128

subclass_column = args.stratification
split_path = "train_test_splits/Nodule_Level_30Splits/nodule_split_1.csv"
subclass_path = 'subclass_labels/subclasses.csv'
feature_path = 'LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv'
results_root_dir = 'test_results/'
test_name = args.test_name
verbose = args.verbose

num_subclasses = 4
subtypes = ["Overall"]
if subclass_column == 'cluster':
    subtypes.extend(["Predominantly Benign", "Somewhat Benign", "Somewhat Malignant", "Predominantly Malignant"])
elif subclass_column == 'spic_groups':
    subtypes.extend(["Unspiculated benign", "Spiculated benign", "Spiculated malignant", "Unspiculated malignant"])
elif subclass_column == 'malignancy':
    subtypes.extend(["Highly Unlikely", "Moderately Unlikely", "Moderately Suspicious", "Highly Suspicious"])
else:
    subtypes.extend(list(range(num_subclasses)))


if args.e2e:

    erm_class = ERMLoss
    erm_args = [None, torch.nn.CrossEntropyLoss()]
    gdro_class = GDROLoss
    gdro_args = [None, torch.nn.CrossEntropyLoss(), eta, num_subclasses]
    optimizer_args_erm = {'lr': lr_erm, 'weight_decay': wd_erm}
    optimizer_args_dro = {'lr': lr_dro, 'weight_decay': wd_dro}
    results = {"Accuracies": {}, "Accuracies_Train": {},"Accuracies_Validation": {}}
    BEST_EPOCH_RESULTS = {"Best_Test_Accuracy": {}}

    for loss_class, loss_args in zip([erm_class, gdro_class], [erm_args, gdro_args]):

        fn_name = loss_class.__name__

        if verbose:
            print(f"Running trials: {fn_name}")

        accuracies, accuracies_train,accuracies_val,best_test_accuracies = run_trials_images(
        
            epochs=epochs,
            csv_files = glob.glob(os.path.join('./data/train_test_splits/Nodule_Level_30Splits/', "*.csv")),
            fn_name = fn_name,
            batch_size = batch_size,
            subclass_column = subclass_column,
            model_class=model_class,
            model_args=model_args,
            loss_class=loss_class,
            loss_args=loss_args,
            optimizer_class=optimizer_class,
            optimizer_args_erm=optimizer_args_erm,
            optimizer_args_dro=optimizer_args_dro,
            device=device,
            verbose=verbose,
            record=True,
        
        )
        results["Accuracies"][fn_name] = accuracies
        results["Accuracies_Train"][fn_name] = accuracies_train 
        results["Accuracies_Validation"][fn_name] = accuracies_val
        BEST_EPOCH_RESULTS["Best_Test_Accuracy"][fn_name] = best_test_accuracies

if args.designed:

    df = data_utils.preprocess_data(
        *data_utils.load_lidc(
            data_root='data/',
            feature_path=feature_path,
            subclass_path=subclass_path
        ),
        subclass_column=subclass_column
        )
    
    erm_class = ERMLoss
    erm_args = [None, torch.nn.CrossEntropyLoss()]
    gdro_class = GDROLoss
    gdro_args = [None, torch.nn.CrossEntropyLoss(), eta, num_subclasses]

    optimizer_args_erm = {'lr': lr_erm, 'weight_decay': wd_erm}
    optimizer_args_dro = {'lr': lr_dro, 'weight_decay': wd_dro}

    results = {"Accuracies": {}, "Accuracies_Train": {},"Accuracies_Validation": {}}
    BEST_EPOCH_RESULTS = {"Best_Test_Accuracy": {}}

    for loss_class, loss_args in zip([erm_class, gdro_class], [erm_args, gdro_args]):
        fn_name = loss_class.__name__
        if verbose:
            print(f"Running trials: {fn_name}")
        
        accuracies, accuracies_train,accuracies_val,best_test_accuracies = run_trials_designedFeatures(
            epochs=epochs,
            df = df,
            csv_files = glob.glob(os.path.join('./data/train_test_splits/Nodule_Level_30Splits/', "*.csv")),
            fn_name = fn_name,
            batch_size = batch_size,
            model_class=model_class,
            model_args=model_args,
            loss_class=loss_class,
            loss_args=loss_args,
            optimizer_class=optimizer_class,
            optimizer_args_erm=optimizer_args_erm,
            optimizer_args_dro=optimizer_args_dro,
            device=device,
            verbose=verbose,
            record=True,
            num_subclasses=num_subclasses
        )
        results["Accuracies"][fn_name] = accuracies
        results["Accuracies_Train"][fn_name] = accuracies_train 
        results["Accuracies_Validation"][fn_name] = accuracies_val
        BEST_EPOCH_RESULTS["Best_Test_Accuracy"][fn_name] = best_test_accuracies


    
accuracies_df_test = pd.DataFrame(
results["Accuracies"],
index=pd.MultiIndex.from_product(
    [range(30), range(epochs), subtypes],
    names=["trial", "epoch", "subtype"]
    )
    )

accuracies_df_train = pd.DataFrame(
results["Accuracies_Train"],
index=pd.MultiIndex.from_product(
    [range(30), range(epochs), subtypes],
    names=["trial", "epoch", "subtype"]
    )
    )


accuracies_df_Validation = pd.DataFrame(
results["Accuracies_Validation"],
index=pd.MultiIndex.from_product(
    [range(30), range(epochs), subtypes],
    names=["trial", "epoch", "subtype"]
    )
    )

accuracies_BEST_df = pd.DataFrame(
BEST_EPOCH_RESULTS["Best_Test_Accuracy"],
index=pd.MultiIndex.from_product(
    [range(30),subtypes],
    names=["trial", "subtype"]
    )   
    )

now = datetime.now()
results_dir = results_root_dir + f'{test_name}_{now.strftime("%Y%m%d_%H%M%S")}/'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

accuracies_df_test.to_csv(results_dir + f'test_accuraciesPerEpoch.csv')
accuracies_df_train.to_csv(results_dir + f'train_accuraciesPerEpoch.csv')
accuracies_df_Validation.to_csv(results_dir + f'validation_accuraciesPerEpoch.csv')
accuracies_BEST_df.to_csv(results_dir + f'BEST_accuracies.csv')


if args.colab:
    from google.colab import files
    files.download(results_dir + f'accuracies.csv')
    files.download(results_dir + f'q.csv')
    files.download(results_dir + f'roc.csv')