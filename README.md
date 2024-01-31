# PASCL
Source code for "PASCL: Supervised Contrastive Learning with Perturbative Augmentation for Particle Decay Reconstruction"

We propose a novel supervised graph contrastive learning with
perturbation augmentation, which utilizes a graph neural network to extract
semantic features from the momentum and energy of particles for the particle decay
reconstruction task.

The code structure is based in part on [BaumBauen](https://github.com/Helmholtz-AI-Energy/BaumBauen). (Thanks for their awesome work.)

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
### Run
To train the PASCL, run the script ```scripts/training/train_model.py``` as follows:
```
cd scripts/training
python train_model.py -c config_yaml_name
```
Note that you should modify your own ```config_yaml_name``` file as input.
