# QPRel

The repository contains code implementing the QPRel approach for certification of the NN classifiers.

## Installation

The instrucions below are given assuming you are working in a conda environment with python 3.7 or later called `myenv`.

1. Gurobi, obtain a license and install according to the instructions from gurobi.com.

2. Pytorch, install pytorch according to the instructions from pytorch.org (`torchaudio` not required).

3. Install `convex_adversarial` according to the instructions from github.com/locuslab/convex_adversarial.

4. Install python packages from `environment.yml`
```
conda env update --name myenv --file environment.yml
```

5. Install `rdl` package from this repository:
```
conda activate myenv  # activate conda environment myenv
pip install -e .      # install the package using pip
```

## Example

```
import rdl

# load the network
net = rdl.io.load_net(rdl.PATH_MODELS / "net_mnist_8_50.m", architecture_type="mnist_model_8_50")

# load the MNIST dataset
dataset = rdl.dataset.MNIST()

# compute the bounds (for 6 samples only)
## set scip_each=0 to go through the full dataset
## this example uses n_gamma=4 for faster computations (default 10)
results = rdl.dbqp.apply_dbqp(dataset, net, prefix="example", scip_each=-10000, verbose=2, n_gamma=4)

# extract the lower bounds in a pandas.DataFrame
bounds_df, _, _ = rdl.io.read_results_QPRel_pandas(res_list=results)
```