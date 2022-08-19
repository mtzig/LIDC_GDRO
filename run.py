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
parser.add_argument('subclass_column')
parser.add_argument('--test_name', default='test')
parser.add_argument('--cnn', action='store_true')
parser.add_argument('--e2e', action='store_true')
parser.add_argument('--designed', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--colab', action='store_true')
parser.add_argument('--trials', default=100)


args = parser.parse_args()

# hyperparameters

lr = 0.0005
wd = 0.005
eta = 0.01
gamma = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.e2e:
    model_class = models.TransferModel18
    model_args = [True, False, device]
else:
    model_class = models.NeuralNetwork
    if args.cnn:
        model_args = [512, 64, 36, 2]
    else:
        model_args = [64, 36, 2]
optimizer_class = torch.optim.Adam

trials = int(args.trials)
epochs = 100
batch_size = 128
split_path = "data/train_test_splits/LIDC_data_split_old.csv"
subclass_path = 'subclass_labels/subclasses.csv'
subclass_column = args.subclass_column
feature_path = 'LIDC_20130817_AllFeatures2D_MaxSlicePerNodule_inLineRatings.csv'

results_root_dir = 'test_results/'

test_name = args.test_name

verbose = args.verbose

if args.cnn:
    train, val, test = image_data_utils.get_features(device=device, subclass=subclass_column)

elif args.e2e:
    train, val, test = image_data_utils.get_features(device=device, images=True, subclass=subclass_column)

else:
    df = data_utils.preprocess_data(
        *data_utils.load_lidc(
            data_root='data/',
            feature_path=feature_path,
            subclass_path=subclass_path
        ),
        subclass_column=subclass_column
    )

    train, val, test = data_utils.train_val_test_datasets(df, split_path=split_path)




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
# dynamic_class = DynamicLoss
# dynamic_args = [None, torch.nn.CrossEntropyLoss(), eta, gamma, num_subclasses]
# dynamic_soft_args = [None, torch.nn.CrossEntropyLoss(), eta, gamma, num_subclasses, None, torch.nn.Softmax(dim=0)]
# upweight_class = UpweightLoss
# upweight_args = [None, torch.nn.CrossEntropyLoss(), num_subclasses]

optimizer_args = {'lr': lr, 'weight_decay': wd}

results = {"Accuracies": {}, "q": {}, "g": {}, "ROC": {}}

for loss_class, loss_args in zip([erm_class, gdro_class], [erm_args, gdro_args]):
# for loss_class, loss_args in zip([dynamic_class], [dynamic_args]):
    fn_name = loss_class.__name__

    tr = SubclassedDataset(*train)

    if fn_name == erm_class.__name__:
        train_dataloader = InfiniteDataLoader(tr, batch_size)
    else:
        train_dataloader = InfiniteDataLoader(tr, batch_size, weights=image_data_utils.get_sampler_weights(tr.subclasses))


    if verbose:
        print(f"Running trials: {fn_name}")

    accuracies, q_data, roc_data = run_trials(
        num_trials=trials,
        epochs=epochs,
        train_dataloader=train_dataloader,
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
    results["q"][fn_name] = q_data
    results["ROC"]["labels"] = roc_data[1].tolist()
    results["ROC"][fn_name] = roc_data[0].tolist()

accuracies_df = pd.DataFrame(
    results["Accuracies"],
    index=pd.MultiIndex.from_product(
        [range(trials), range(epochs), subtypes],
        names=["trial", "epoch", "subtype"]
    )
)
q_df = pd.DataFrame(
    results["q"],
    index=pd.MultiIndex.from_product(
        [range(trials), range(epochs), subtypes[1:]],
        names=["trial", "epoch", "subtype"]
    )
)

roc_df = pd.DataFrame(results["ROC"])

now = datetime.now()

results_dir = results_root_dir + f'{test_name}/'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

accuracies_df.to_csv(results_dir + f'accuracies.csv')
q_df.to_csv(results_dir + f'q.csv')
roc_df.to_csv(results_dir + f'roc.csv', index=False)

if args.colab:
    from google.colab import files
    files.download(results_dir + f'accuracies.csv')
    files.download(results_dir + f'q.csv')
    files.download(results_dir + f'roc.csv')
