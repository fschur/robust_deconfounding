[![Build Status](https://app.travis-ci.com/fschur/robust_deconfounding.svg?token=pYfyy5csz8JR86HGpneh&branch=main)](https://travis-ci.com/fschur/robust_deconfounding)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2002.05551-b31b1b.svg)](https://arxiv.org/abs/)

# DecoR: Deconfounding Time Series with Robust Regression
This repository provides source code for the paper 
[*DecoR: Deconfounding Time Series with Robust Regression*](https://arxiv.org/). 
In particular, the **robust-deconfounding** package implements robust deconfounding method **DecoR** and the robust linear 
regression methods **Torrent** and **BFS**.

The **experiments** directory holds code for synthetic experiments and provides the necessary scripts to reproduce 
the experimental results reported in the paper.

## Installation
To install the minimal dependencies needed to use the meta-learning algorithms, run in the main directory of this 
repository
```bash
pip install .
``` 

For full support of all scripts in the repository, for instance to reproduce the experiments, further dependencies need
to be installed. 
To do so, please run in the main directory of this repository 
```bash
pip install -r requirements.txt
``` 

## Usage
The following code snippet demonstrates the core functionality of the meta-learners provided in this repository. 
The code in **demo.ipynb** demonstrates the core functionality of the DecoR provided in this repository.


## Reproducing the experiments
Below we point to the experiment scripts that were used to generate the results reported in the paper.

### Synthetic experiments

To run the experiments:

```bash
python experiments/experiments.py

``` 
To run the ablation experiments:

```bash
python experiments/experiments_ablation.py
``` 
You can adjust method number of datapoints and the data generating process in the files themselves.
Note that the experiments may take a long time to run, depending on the number of repetitions and the size of the
datasets especially if **BFS** is used. 
