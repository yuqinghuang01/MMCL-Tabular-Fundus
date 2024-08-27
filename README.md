# Cardiovascular Disease Prediction through Retinal Graph Representations and Multimodal Self-supervised Learning

This is the official code repository for our paper "Cardiovascular Disease Prediction through Retinal Graph Representations and Multimodal Self-supervised Learning". The code is developed upon the [MMCL-Tabular-Imaging](https://github.com/paulhager/MMCL-Tabular-Imaging) framework.

<p align="center">
  <img src="./overview.png?raw=true">
</p>

## Instructions

Run the following commands to install and activate the environment. Then install additional packages.
```
conda env create --file environment.yaml
conda activate selfsuper
pip install torch-geometric
pip install --no-index torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
``` 

To run, execute `python run.py`.

### Data

The UK Biobank data is semi-private. You can apply for access [here](https://www.ukbiobank.ac.uk/enable-your-research/apply-for-access). 

Dataset file paths are specified in `configs/dataset/[dataset].yaml`, where `[dataset]` corresponds to the argument value specified in `configs/config.yaml`. 
- For the images, provide a .pt with a list of your image tensors. Note that images can be raw fundus images or probabilistic vessel maps depending on the strategy.
- For the graphs, provide a .pt with a list of your graphs, in the datatype of `torch_geometric.data`. Vessel graphs can be generated using the Voreen-based tool [OCTA-graph-extraction](https://github.com/KreitnerL/OCTA-graph-extraction).
- For tabular data and labels, files are in `.csv` format.
- The tabular data should be provided as *NOT* one-hot encoded so the sampling from the empirical marginal distribution works correctly. You must provide a file `field_lengths_tabular` which is an array that in the order of your tabular columns specifies how many options there are for that field. Continuous fields should thus be set to 1 (i.e. no one-hot encoding necessary), while categorical fields should specify how many columns should be created for the one_hot encoding  

### Arguments - Command Line

If pretraining, pass `pretrain=True` and `datatype={imaging_tabular|graph_tabular}` for the desired pretraining type, with image encoder or with graph encoder.

If you do not pass `pretrain=True`, the model will train fully supervised with the data modality specified in `datatype`.

You can evaluate a model by passing the path to the final pretraining checkpoint with the argument `checkpoint={PATH_TO_CKPT}`. After pretraining, a model will be evaluated with the default settings (frozen eval, lr=1e-3).

### Arguments - Hydra

All argument defaults can be set in hydra yaml files found in the configs folder.

Code is integrated with weights and biases, so set `wandb_project` and `wandb_entity` in [configs/config.yaml](configs/config.yaml).
