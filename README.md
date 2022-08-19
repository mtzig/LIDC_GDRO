# LIDC_GDRO
Improving Worst-Class Performance in Lung Nodule Malignancy Classification

Thomas Zeng & Elias Furst

For questions: contact Thomas Zeng at zengt@carleton.edu.

## Project Description

Machine learning models have been widely used in lung cancer computer-aided diagnosis (CAD) studies. However, the heterogeneity in the visual appearance of lung nodules as well as lack of consideration of hidden subgroups in the data are significant obstacles to generate accurate CAD outcomes across all nodule instances.  Previous lung cancer CAD models aim to achieve Empirical Risk Minimization (ERM), which leads to a high overall accuracy but often fails at predicting certain subgroups caused by the lung cancer heterogeneity. In this study, we discovered semantically meaningful and hidden lung nodule subgroups and developed a CAD model, utilizing Group Distributionally Robust Optimization (gDRO), that is robust across all subgroups.

Dataloader code is adapted from https://github.com/facebookresearch/DomainBed


## Code Usage

To run our tests use run.py.

As an example the following group will run 10 trials of ERM and gDRO on designed features with spiculation subgroups. The results will be saved in 'spic_designed.csv' in test_results directory.

```
python run.py spic_groups --designed --test_name spic_designed --verbose --trials 10
```

The different stratification methods are `spic_groups`, `cluster`, `malignancy`.

The different features are `--designed`, `--e2e` (image end to end), `--cnn` (deep features)

## License

This source code is released under the MIT license, included [here](LICENSE).
