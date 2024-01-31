# PASCL
Source code for "PASCL: Supervised Contrastive Learning with Perturbative Augmentation for Particle Decay Reconstruction"


## Data

The dataset is [available on Zenodo](https://zenodo.org/records/6983258).

## Usage
### Environment Setup
```
conda create -yn pascl python=3.7
conda activate pascl
#update pip
pip3 install -U pip

git clone https://github.com/lukSYSU/PASCL.git
cd PASCL
pip install .
```
### RUN
To train the PASCL, run the script ```scripts/training/train_model.py``` as follows:
```
cd scripts/training
python train_model.py -c config_yaml_name
```
Note that you should modify your own ```config_yaml_name``` file as input.
