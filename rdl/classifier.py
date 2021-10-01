# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as tud

import rdl
import rdl.dataset as ds
import rdl.utils as utils

import convex_adversarial as ca




# >>>>>>>> This part is borrowed from
# https://github.com/locuslab/convex_adversarial/
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def mnist_model(): 
    model = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7, 100),
        nn.ReLU(), 
        nn.Linear(100, 10)
    )
    return model


def mnist_model_small(): 
    model = nn.Sequential(
        nn.Conv2d(1, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(16*7*7, 100),
        nn.ReLU(), 
        nn.Linear(100, 10)
    )
    return model


def mnist_model_large(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(64*7*7, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    return model


def model_wide(in_ch=1, out_width=7, k=4):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4*k, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*k, 8*k, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(8*k*out_width*out_width, k*128),
        nn.ReLU(),
        nn.Linear(k*128, 10)
    )
    return model


def mnist_model_1_300(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28, 300),
        nn.ReLU(),
        nn.Linear(300, 10), # 1
    )
    return model


def mnist_model_1_50(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28, 50),
        nn.ReLU(),
        nn.Linear(50, 10), # 1
    )
    return model


def mnist_model_2_50(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28, 50),
        nn.ReLU(),
        nn.Linear(50, 50), # 1
        nn.ReLU(),
        nn.Linear(50, 10)  # 2
    )
    return model


def mnist_model_2_50_(): 
    model = nn.Sequential(
        nn.Linear(28*28, 50),
        nn.ReLU(),
        nn.Linear(50, 50), # 1
        nn.ReLU(),
        nn.Linear(50, 10)  # 2
    )
    return model


def mnist_model_3_50_(): 
    model = nn.Sequential(
        nn.Linear(28*28, 50),
        nn.ReLU(),
        nn.Linear(50, 50), # 1
        nn.ReLU(),
        nn.Linear(50, 50), # 2
        nn.ReLU(),
        nn.Linear(50, 10)  # 3
    )
    return model

def mnist_model_4_50(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28, 50),
        nn.ReLU(),
        nn.Linear(50, 50), # 1
        nn.ReLU(),
        nn.Linear(50, 50), # 2
        nn.ReLU(),
        nn.Linear(50, 50), # 3
        nn.ReLU(),
        nn.Linear(50, 10)  # 4
    )
    return model


def mnist_model_8_50(): 
    model = nn.Sequential(
        Flatten(),
        nn.Linear(28*28,50),
        nn.ReLU(),
        nn.Linear(50, 50), # 1
        nn.ReLU(),
        nn.Linear(50, 50), # 2
        nn.ReLU(),
        nn.Linear(50, 50), # 3
        nn.ReLU(),
        nn.Linear(50, 20), # 4
        nn.ReLU(),
        nn.Linear(20, 20), # 5
        nn.ReLU(),
        nn.Linear(20, 20), # 6
        nn.ReLU(),
        nn.Linear(20, 20), # 7
        nn.ReLU(),
        nn.Linear(20, 10)  # 8
    )
    return model


def other_model(n_input, n_output, n_hidden=2):
    model = SequentialNet(
        sum(
            [(nn.Linear(10, 10), nn.ReLU()) for _ in range(n_hidden-1)],
            (Flatten(), nn.Linear(n_input, 10), nn.ReLU())
        ) +
        (nn.Linear(10, n_output), )
    )
    return model


def other_model_2_50():
    model = nn.Sequential(
        Flatten(),
        nn.Linear(24, 50),
        nn.ReLU(),
        nn.Linear(50, 50), # 1
        nn.ReLU(),
        nn.Linear(50, 2)  # 2
    )
    return model
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




class SequentialNet(nn.Sequential):
    """ nn.Module for a feed-forward neural net with dense or convolutional layers and ReLU activation.

    Attributes
    ----------
    n_features : int
        Number of input features.
    n_layers : int
        Number of dense layers.
    n_classes : int
        Number of output classes.
    size_list : list of int
        List of length n_layers+1 (including the input layer) containing numbers of nodes in
        corresponding layers.
    layers : nn.ModuleList
        Module of length n_layers containing dense layers as torch.nn.Linear objects.


    Methods
    -------
    forward
        Computes output of the NN given input X.
    copy_structure
        Constructs new DenseNet object with the same attributes and architecture,
        note that initial weights/biases of the new NN will be different.
    init_global
        Initialization of the weights/biases in the NN using provided (in place!) initializer from nn.init.
    labels
        Computes the predicted labels.
    ccr
        Computes the correct classification rate given features and true labels.

    """


    def __init__(self, layers_list,
                 input_size=None, n_features=None, n_layers=None, n_classes=None, size_list=None,
                 device=None):
        """ Constructor, defines the layers of the NN.

        """
        super().__init__(*layers_list)
        self.layers = [l for l in self if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d)]

        if input_size is None:
            if isinstance(self[0], nn.Linear):
                input_size = [self[0].in_features]
            elif isinstance(self[0], Flatten) and isinstance(self[1], nn.Linear):
                input_size = [self[1].in_features]
            else:
                raise ValueError("Found not supported layer type.")

        self.input_size = input_size

        if size_list is None:
            size_list = self.set_size_list()
        else:
            self.size_list = size_list

        self.neurons_list = [np.prod(s) for s in self.size_list]

        # set default values
        self.n_features = n_features if n_features is not None else np.prod(size_list[0])
        self.n_layers = n_layers if n_layers is not None else len(size_list) - 1
        self.n_classes = n_classes if n_classes is not None else np.prod(size_list[-1])

        # put the NN on GPU if any available
        if device is None:
            device = torch.device("cpu")

        self.to(device)
        return


    def to(self, device, **kwargs):
        """ Update the .device attribute in addition to the usual functionality.

        """
        self.device = device
        return super().to(device, **kwargs)


    def __repr__(self):
        """ Prints the architecture and initializers of the NN.

        """
        r = ""
        r += super().__repr__()
        r += "\n"
        return r


    def set_size_list(self):
        """ Passes a zero-tensor through the network to get all the intermediate dimensions.

        """
        self.size_list = [self.input_size]

        x = torch.zeros(1, *self.input_size)
        for layer in self:
            x = layer(x)
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                self.size_list.append(x.shape[1:])

        return self.size_list


    def show_state_dict(self):
        """ Prints the keys and values' shape from the state dict.

        """
        utils.show_state_dict(self.state_dict())
        return


    def copy_structure(self):
        """ Makes a new NN with the same architecture.

        """
        net_copy = SequentialNet(*self, size_list=self.size_list)
        return net_copy


    def labels(self, X, numpy=False):
        """ Computes the predicted labels.

        """
        if isinstance(X, np.ndarray):
            X = torch.Tensor(X).to(self.device)

        if isinstance(X, torch.Tensor):
            X = X.view(-1, *self.input_size).to(self.device)

            y_logits = self(X)
            if not numpy:
                return y_logits.argmax(dim=1)
            else:
                return np.array(y_logits.argmax(dim=1), dtype=int)

        if isinstance(X, ds.Dataset):
            out = np.array([]) if numpy else torch.LongTensor([]).to(self.device)
            for X_batch, _ in X.loader:
                labels = self.labels(X_batch)
                if numpy:
                    out += labels
                else:
                    out = torch.cat((out, labels))
            return out


    def ccr(self, *args):
        """ Computes the correct classification rate given features and true labels.

        """
        # case 0: Dataset
        if len(args)==1 and isinstance(args[0], ds.Dataset_):
            X = args[0].data
            y_true = args[0].targets.to(self.device)
        elif len(args)==1 and isinstance(args[0], tud.Dataset):
            X = args[0]
            y_true = X.targets.to(self.device)
        # case 1: samples and labels
        elif len(args)==2:
            X, y_true = args
        # case 2: invalid input
        else:
            raise ValueError(f"Invalid input(s) {[type(arg) for arg in args]}, should be either a single Dataset or samples and labels.")

        return utils.ccr(self.labels(X), y_true).item()


    @classmethod
    def get_weight_matrix(cls, layer, input_size=None, bias=False, to_numpy=False):
        """ Get weight matrix and bias vector from a linear layer (usually dense or convolutional).

        """
        if isinstance(layer, nn.Linear):
            if not bias:
                return \
                    layer.weight if not to_numpy else \
                    layer.weight.detach().cpu().numpy()
            else:
                return \
                    [layer.weight,
                     layer.bias] if not to_numpy else \
                    [layer.weight.detach().cpu().numpy(),
                     layer.bias.detach().cpu().numpy()]

        elif isinstance(layer, nn.Conv2d):
            if not hasattr(layer, "converted_to_linear"):
                if input_size is None:
                    raise ValueError("input_size is required for convolutional layers (got None).")

                layer.converted_weight = utils.convert_cnn_to_dense(layer, input_size)
                layer.converted_to_linear = True

            out = layer.converted_weight[0] if not bias else layer.converted_weight
            return out if not to_numpy else utils.to_numpy(out)
        
        else:
            raise NotImplementedError(f"get_weight_matrix supports only nn.Linear and nn.Conv2d, got {type(layer)}.")


    def iter_weights(self, return_generator=False, bias=False, drop_last=False, to_numpy=False):
        """ Generator for the weights of the dense layers as np.arrays.

        """
        weights = (SequentialNet.get_weight_matrix(layer, input_size=self.size_list[i], bias=bias, to_numpy=to_numpy)
                   for i, layer in enumerate(self.layers if not drop_last else self.layers[:-1]))

        return weights if return_generator else tuple(weights)


    def iter_weight_norms(self, return_generator=False, drop_last=False, to_numpy=False):
        """ Generator for the squared (!!!) spectral norms of the weight matrices.

        """
        if to_numpy:
            norms = (
                np.linalg.norm(W @ W.T, ord=2) for
                W in self.iter_weights(return_generator=True, drop_last=drop_last, to_numpy=True))
            return norms if return_generator else np.fromiter(norms, dtype=float)

        else:
            with torch.no_grad():
                norms = (
                    utils.largest_eigenvalue(W @ W.T) for
                    W in self.iter_weights(return_generator=True, drop_last=drop_last, to_numpy=False))
            return norms #if return_generator else torch.Tensor(norms, dtype=float)


    def to_kw_sequential(self):
        """ Returns the corresponding net of KWSequential type.

        """
        netKW = KWSequential(self.size_list)
        netKW.load_state_dict_from_dense_net(self.state_dict())

        return netKW


    def get_activation_bounds(self, x_anchor, epsilon, norm_type):
        """ Uses convex_adversarial package to construct bounds on pre-ReLU activations from input bounds.
        
        """
        
        kwnet_dual = ca.DualNetwork(
            self,
            x_anchor.view(-1, *self.input_size),
            norm_type={"2":"l2", "i":"l1"}[norm_type], epsilon=epsilon)

        box_bounds_activation = []
        for layer in kwnet_dual.dual_net:
            if isinstance(layer, ca.dual_layers.DualReLU):
                box_bounds_activation.append(
                    list(
                        zip(
                            layer.zl.detach().numpy().reshape(-1),
                            layer.zu.detach().numpy().reshape(-1))
                    )
                )

        return box_bounds_activation