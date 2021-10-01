# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as spl
import scipy.sparse as ssp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import Counter

import itertools
from time import time

from rdl import PATH_RESULTS

import logging
import warnings

inf = float("inf")


def suffix_from_specs(**kwargs):
    net_name = kwargs["net_name"]
    net_name = net_name if isinstance(net_name, str) else f"[{net_name[0]}, {net_name[1]}]"

    SUFFIX =  f"net_name-{kwargs['net_name']}-"
    SUFFIX += f"data_name-{kwargs['data_name']}-"
    SUFFIX += f"weight_qp-{kwargs['weight_qp']}-"
    SUFFIX += f"n_epochs-{kwargs['n_epochs']}-"
    SUFFIX += f"pretrain_epochs-{kwargs['pretrain_epochs']}-"
    SUFFIX += f"max_time_opt-{kwargs['max_time_opt']}"
    return SUFFIX


def to_numpy(args):
    """ Convert arguments to numpy arrays.

    """
    if isinstance(args, list) or isinstance(args, tuple):
        return [arg if isinstance(arg, np.ndarray) else arg.detach().cpu().numpy() for arg in args]
    elif isinstance(args, np.ndarray):
        return args
    elif isinstance(args, torch.Tensor):
        return args.detach().cpu().numpy()
    else:
        raise ValueError(f"Wrong input type, got {type(args)}, supported are tuple, list, np.ndarray and torch.Tensor.")


def cholesky_block_tridiagonal(main_diag_blocks, sub_diag_blocks, lower=True, first_diag_block=False):
    """ Cholesky of a tridiagonal block matrix.

    See:
    ----
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.16.7755&rep=rep1&type=pdf

    """
    if not lower:
        raise NotImplementedError("cholesky_block_tridiagonal does not support the upper cholesky decomposition.")

    main_diag_blocks_out = []
    sub_diag_blocks_out = []
    for i, (D, B) in enumerate(itertools.zip_longest(main_diag_blocks, sub_diag_blocks)):
        # diagonal block
        if i == 0:
            if first_diag_block:
                L = torch.diag(D.diag().sqrt())
            else:
                L = torch.cholesky(D, upper=False)
        else:
            L = torch.cholesky(D - C @ C.T, upper=False)

        main_diag_blocks_out.append(L)
        if B is None: # last iteration: update only the diagonal block
            break

        # sub-diagonal blocks
        C, _ = torch.triangular_solve(B.T, L, upper=False)
        C = C.T
        sub_diag_blocks_out.append(C)

    return main_diag_blocks_out, sub_diag_blocks_out


def random_blocks(n_list=None, diag=False, device=None):
    """ Creates random blocks (diagonal and sub-diagonal) for testing.

    """
    if device is None:
        device = torch.device("cpu")
    if n_list is None:
        n_list = [784, 50, 50, 50, 50, 10]

    main_diag_blocks = []
    sub_diag_blocks = []

    # generate random blocks
    for n, m in zip(n_list[:-1], n_list[1:]):
        main_diag_blocks.append(
            torch.diag(torch.rand(n, device=device)) if diag else torch.tril(torch.rand(n, n, device=device)))
        sub_diag_blocks.append(
            torch.rand(m, n, device=device))

    n = n_list[-1]
    main_diag_blocks.append(
        torch.diag(torch.rand(n, device=device)) if diag else torch.tril(torch.rand(n, n, device=device)))

    return main_diag_blocks, sub_diag_blocks


def cholesky_block_tridiagonal_test(n_list=None, diag=False, device=None):
    """ Test for cholesky_block_tridiagonal.

    """
    if device is None:
        device = torch.device("cpu")
    if n_list is None:
        n_list = [784, 50, 50, 50, 50, 10]

    print("Test cholesky_block_tridiagonal for the following block sizes:")
    print(n_list)
    main_diag_blocks, sub_diag_blocks = random_blocks(n_list, diag, device)

    # TEST
    main_diag_blocks = [M_ + 10*torch.eye(len(M_), device=device) for M_ in main_diag_blocks]

    # assemble L
    L_true = assemble_from_tridiagonal_blocks(main_diag_blocks, sub_diag_blocks, symmetric=False)

    # compute M block-wise
    main_diag_blocks_M = [main_diag_blocks[0] @ main_diag_blocks[0].T]
    sub_diag_blocks_M = []
    for L_old_, L_, C_ in zip(main_diag_blocks[:-1], main_diag_blocks[1:], sub_diag_blocks):
        sub_diag_blocks_M.append(C_ @ L_old_.T)
        main_diag_blocks_M.append(C_ @ C_.T + L_ @ L_.T)

    M = assemble_from_tridiagonal_blocks(main_diag_blocks_M, sub_diag_blocks_M, symmetric=True)

    # compare to L @ L.T
    print("Error M", torch.dist(M, L_true @ L_true.T))

    t = time()
    #L_chol = torch.cholesky(L_true @ L_true.T, upper=False)
    L_chol = torch.cholesky(M, upper=False)
    print(time() - t)

    print("Error L (true vs torch)", torch.dist(L_chol, L_true))

    t = time()
    L_copm = assemble_from_tridiagonal_blocks(
        *cholesky_block_tridiagonal(main_diag_blocks_M, sub_diag_blocks_M, first_diag_block=diag),
        symmetric=False)
    print(time() - t)

    print("Error L (true vs rdl)", torch.dist(L_copm, L_true))
    print("Error L (torch vs rdl)", torch.dist(L_copm, L_chol))


def assemble_from_tridiagonal_blocks(main_diag_blocks, sub_diag_blocks, symmetric=True):
    """ Given D1, D2, ..., DN and B1, ..., BN-1 returns

    D1  B1.T
    B1  D2   B2.T
        B2   ...  ...
             ...  DN-1  BN-1.T
                  BN-1  DN

        or 

    D1  0
    B1  D2   0
        B2   ...  ...
             ...  DN-1  0
                  BN-1  DN

    """
    DD = torch.block_diag(*main_diag_blocks)
    BB = torch.block_diag(*sub_diag_blocks)

    DD[DD.shape[0] - BB.shape[0]:, :BB.shape[1]] += BB
    if symmetric:
        DD[:BB.T.shape[0], DD.shape[1] - BB.T.shape[1]:] += BB.T

    return DD


def check_box_bounds_activation(net, x_anchor, norm_bounds_input, box_bounds_activation, test_batch=100, norm="2"):
    """
    
    """
    if norm_bounds_input is None:
        return

    if norm == "i":
        noise = norm_bounds_input * torch.rand(test_batch, *x_anchor.shape)
    elif norm == "2":
        d = np.prod(x_anchor.shape)
        U = torch.randn(test_batch, *x_anchor.shape)
        U = torch.diag(1.0 / U.view(test_batch, -1).norm(p=2, dim=1)) @ U
        noise = norm_bounds_input * torch.diag(torch.rand(test_batch)**(1/d)) @ U

    else:
        raise ValueError(f"Given norm ({norm}) is not supported.")

    l = 0
    x = noise + x_anchor.expand(test_batch, *x_anchor.shape)
    for layer in net:
        if l == len(box_bounds_activation):
            break

        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            x = layer(x)
            n = np.prod(x.shape[1:])
            lb = torch.Tensor(box_bounds_activation[l])[:, 0].expand(test_batch, n)
            ub = torch.Tensor(box_bounds_activation[l])[:, 1].expand(test_batch, n)
            assert (lb <= x.view(x.size(0), -1)).all() & (x.view(x.size(0), -1) <= ub).all()
            l += 1
            x = nn.functional.relu(x)
        else:
            x = layer(x)

    return


def projection_matrix(I):
    """ Creates a projection matrix from a bool vector.
    
    """
    if not isinstance(I, np.ndarray):
        I = np.array(I)
    M = ssp.coo_matrix((np.full(I.sum(), 1.0),(np.arange(I.sum()), np.arange(len(I))[I]))) # (data, (row, column))

    return M


def convert_cnn_to_dense(layer_cnn, input_size, test_batch=0, test_threshold=1e-6):
    """ Transforms a CNN layer for a given input size to an equivalent dense linear layer.

    """
    device = layer_cnn.weight.device # do everything on the device of the given conv layer
    x = [(i,)+l for i, l in enumerate(itertools.product(*[range(s) for s in input_size]))]

    with torch.no_grad():
        i = torch.LongTensor(x)
        v = torch.ones(len(i))

        input_tensor = torch.sparse.LongTensor(i.transpose(0, 1), v, torch.Size([len(i)] + list(input_size))).to(device)
        output_tensor = layer_cnn(input_tensor.to_dense())
        output_size = output_tensor.size()[1:]

        bias_mlp = layer_cnn.bias.repeat(int(np.prod(output_size[-2:])), 1).t().reshape(-1)
        weight_mlp = (output_tensor.view(len(i), -1) - bias_mlp).transpose(0, 1)
        weight_mlp.to_sparse()

        if test_batch:
            # just for debugging purposes
            layer_mlp = nn.Linear(np.prod(input_size), np.prod(output_size))
            layer_mlp.weight.data = weight_mlp if isinstance(weight_mlp, torch.Tensor) else weight_mlp.to_dense()
            layer_mlp.bias.data = bias_mlp
            x = torch.randn([test_batch] + input_size)
            output_cnn = layer_cnn(x).view(test_batch, -1)
            output_mlp = layer_mlp(x.view(test_batch, -1))
            assert ((output_cnn - output_mlp).abs() < test_threshold).all()
            print("ok!")

        return weight_mlp, bias_mlp


def unfold_alpha(alpha, neurons_list):
    """ Transforms 1-d array into a list of arrays where each element has length
    equal to the number of neurons in the corresponding layer.

    """
    to_numpy = not isinstance(alpha[0], torch.Tensor)

    if len(alpha) != len(neurons_list)-2:
        raise ValueError(
            f"Length of alpha ({len(alpha)}) has to be " +
            f"length of neurons_list ({len(neurons_list)}) minus 2.")

    if to_numpy:
        return [np.full((n_l, ), alpha_l) for alpha_l, n_l in zip(alpha, neurons_list[1:-1])]
    else:
        return [torch.full((n_l, ), alpha_l.item(), device=alpha_l.device) for alpha_l, n_l in zip(alpha, neurons_list[1:-1])]


def show_state_dict(d):
    """ Prints the keys and values' shape from the state dict.

    """
    for key, val in d.items():
        print(f"{key:20s}", type(val), val.shape)


def get_cd_from_DBQP(qp):
    """

    """
    d = qp.obj["distance_squared"].getValue()
    c = np.array([gap.getValue() for gap in qp.obj["propagation_gaps"]])

    return c, d


def get_AB_from_DenseNet(net):
    """

    """
    M = get_qobj_matrix(net)

    nsum = 0
    for l, nl in enumerate(net.neurons_list[:-1]):
        if l==0:
            A = np.zeros(M.shape)
            A[:nl, :nl] = M[:nl, :nl]
            B = []

        else:
            B.append(np.zeros(M.shape))
            B[-1][nsum:nsum+nl, nsum-nold:nsum] = M[nsum:nsum+nl, nsum-nold:nsum]
            B[-1][nsum-nold:nsum, nsum:nsum+nl] = M[nsum-nold:nsum, nsum:nsum+nl]
            B[-1][nsum:nsum+nl, nsum:nsum+nl] = M[nsum:nsum+nl, nsum:nsum+nl]

        nsum += nl
        nold = nl

    return A, B


def get_alpha_max(net, alpha0=1.0, to_numpy=None):
    """ Given a net with weights W(l) returns the following values for alpha:
               alpha(1) = 4            / ||W(1)||^2 and
               alpha(l) = 4*alpha(l-1) / ||W(l)||^2 for l>1

    """
    device = net.parameters().__next__().device
    if to_numpy is None:
        to_numpy = device.type != "cuda"

    # NOTE: if numpy is True, we get a usual numpy array, otherwise it is still a 
    #       numpy array, but with scalar torch tensors as elements (device preserved).
    if to_numpy:
        alpha = np.cumprod(
            [4/norm for norm in net.iter_weight_norms(to_numpy=to_numpy)]
        )
    else:
        alpha = torch.cumprod(
            torch.Tensor([4/norm for norm in net.iter_weight_norms(to_numpy=to_numpy)]).to(device),
            dim=0
        )

    # NOTE: no ReLU after the last linear transformation
    return unfold_alpha(alpha0*alpha[:-1], net.neurons_list)


def move_alpha_to(alpha, where):
    """ Moves alpha between a torch.Tensor on gpu and a numpy.ndarray.

    """
    if where == "cpu":
        return [alpha_.detach().cpu().numpy() for alpha_ in alpha]
    if where == "gpu":
        return [torch.Tensor(alpha_).to(torch.device("cuda")).detach() for alpha_ in alpha]


def get_alpha_safe(
        net, one_hidden_special_case=True, eps_alpha=1e-4, alpha0=1.0, to_numpy=None):
    """ Given a net with weights W(l) returns the following values for alpha:
               alpha(1) = 2          / ||W(1)||^2 and
               alpha(l) = alpha(l-1) / ||W(l)||^2 for l>1

    """
    device = net.parameters().__next__().device
    if to_numpy is None:
        to_numpy = device.type != "cuda"


    # in case of one hidden layer the optimal alpha is 4 / ||W(1)||^2
    if one_hidden_special_case and len(net.layers) == 2:
        print("NOTE: network with one hidden layer, use the necessary alpha.")
        return [alpha*(1-eps_alpha) for alpha in get_alpha_max(net, alpha0, to_numpy=to_numpy)]

    # NOTE: if numpy is True, we get a usual numpy array, otherwise it is still a 
    #       numpy array, but with scalar torch tensors as elements (device preserved).
    if to_numpy:
        alpha = 2 * np.cumprod(
            [1/norm for norm in net.iter_weight_norms(to_numpy=to_numpy)]
        )
    else:
        alpha = 2 * torch.cumprod(
            torch.Tensor([1/norm for norm in net.iter_weight_norms(to_numpy=to_numpy)]).to(device),
            dim=0
        )

    # NOTE: no ReLU after the last linear transformation
    return unfold_alpha(alpha0*alpha[:-1], net.neurons_list)


def get_alpha_opt(net, alpha_min=None, alpha_max=None, alpha_old=None,
    n_max=100, beta=0.5, precision=1e-4, alpha0=1.0, verbose=0, to_numpy=True):
    """ Performs a binary search between alpha_min and alpha_max.

    We want to start solving QPRel with the largest alpha between
    alpha_min and alpha_max such that the matrix from the quadratic
    objective function is still positive definite. alpha_min is
    usually computed by get_alpha_safe(self.net) and alpha_max by
    get_alpha_max(self.net). Checking whether the matrix is positive
    semi-definite is done using the Cholesky decomposition.

    Parameters:
    -----------
    net : rdl.classifier.DenseNet
        Weights are taken from this network
    alpha : np.array
        Initial safe alpha, array of non-negative multipliers (default alpha=None, i.e. get_alpha_safe(net))
    d : numpy.array
        Direction in which we perform the binary search, will modify alpha_max (default d=None)
    n_max : int
        Maximum number of iterations of the binary search (default n_max=15)
    beta : float in (0,1)
        Determines where to put the new boundary with respect to the length of the current interval (default beta=0.5)
    precision : float
        Search stops if the achieved relative precision is below this value.
    verbose : int
        Verbosity level.

    Output:
    -------
    alpha_min : numpy.array
        Best found alpha value.

    """
    if verbose >= 2:
        print("  [get_alpha_opt]: Find the optimal alpha values")

    # initialize the search interval
    # ------------------------------
    if alpha_old is not None:
        psd = check_psd(get_qobj_matrix(
            net, alpha_old, alpha0=alpha0, to_numpy=to_numpy, return_blocks=not to_numpy))
        if psd:
            if alpha_min is not None:
                warnings.warn("Provided alpha_old overwrites provided alpha_min.")
            alpha_min = alpha_old
            alpha_min_set = True
            alpha_max_set = False
            beta += (0.0 - beta) * 0.5
        else:
            alpha_max = alpha_old
            alpha_min_set = False
            alpha_max_set = True
            beta += (1.0 - beta) * 0.5
    else:
        alpha_min_set, alpha_max_set = False, False

    if alpha_min is None:
        alpha_min = get_alpha_safe(net, alpha0=alpha0, to_numpy=to_numpy)
    elif alpha_min is not None and not alpha_min_set:
        psd = check_psd(get_qobj_matrix(
            net, alpha_min, alpha0=alpha0, to_numpy=to_numpy, return_blocks=not to_numpy))
        if not psd:
            warnings.warn("Given initial alpha unsafe! -> proceed with rdl.utils.get_alpha_safe")
            alpha_min = get_alpha_safe(net, alpha0=alpha0, to_numpy=to_numpy)

    if alpha_max is None:
        alpha_max = get_alpha_max(net, alpha0=alpha0, to_numpy=to_numpy)
    # ------------------------------

    psd = check_psd(get_qobj_matrix(
        net, alpha_max, alpha0=alpha0, to_numpy=to_numpy, return_blocks=not to_numpy))

    if verbose >= 3:
        print("  alpha_min:", alpha_min)
        print("  alpha_max:", alpha_max)

    if psd:
        #print_psd_track(True)
        if not to_numpy: # DEBUGGING
            logging.debug("  given alpha_max leads to a PSD matrix")
        return alpha_max
    else:
        if not to_numpy: # DEBUGGING
            logging.debug("  given alpha_max doesn't lead to a PSD matrix")

    i = 0
    psd_track = []

    psd = check_psd(get_qobj_matrix(
        net, alpha_min, alpha0=alpha0, to_numpy=to_numpy, return_blocks=not to_numpy))
    psd_track.append(psd)

    if verbose >= 2:
        print(f"------- STEP {i} -------")
        if psd:
            print("PSD")
        else:
            warnings.warn("non-PSD -> initial alpha unsafe!")

    def get_gap(a1, a2):
        if to_numpy:
            gap = np.max([
                np.linalg.norm(alpha_max_l - alpha_min_l, ord=2) / np.linalg.norm(alpha_min_l, ord=2) for
                alpha_min_l, alpha_max_l in zip(a1, a2)])
        else:
            gap = max([
                torch.norm(alpha_max_l - alpha_min_l, p=2) / torch.norm(alpha_min_l, p=2) for
                alpha_min_l, alpha_max_l in zip(a1, a2)])
        return gap

    gap = get_gap(alpha_min, alpha_max)

    # start binary search
    while i < n_max and gap > precision:
        i += 1

        alpha_new = [alpha_min_l + beta*(alpha_max_l - alpha_min_l) for
                     alpha_min_l, alpha_max_l in zip(alpha_min, alpha_max)]
        psd = check_psd(get_qobj_matrix(
            net, alpha_new, alpha0=alpha0, to_numpy=to_numpy, return_blocks=not to_numpy))
        psd_track.append(psd)

        if verbose >= 3:
            print(f"------- STEP {i} -------")
            print("PSD -> update alpha_min" if psd else "non-PSD -> update alpha_max")
        if psd:
            alpha_min = alpha_new
        else:
            alpha_max = alpha_new

        gap = get_gap(alpha_min, alpha_max)
        if verbose > 2:
            print(f"Relative gap: {gap:.6f}{' > ' if gap>precision else ' < '}{precision}")

    if verbose >= 2:
        if verbose >= 3:
            print(f"  alpha_opt: {alpha_min}")
        print(f"  iters: {i} / {n_max}")
        print(f"  gap: {gap} <? {precision:.3f}")

    #print_psd_track(psd_track)
    return alpha_min


def print_psd_track(psd_track, width=80):
    """ Binary search result visualization.

    """
    r = "|" + " "*width + "|"
    if isinstance(psd_track, bool):
        r = r[:-2] + "o" + r[-1]
        print(r)
        return

    r = r[:-2] + "x" + r[-1]
    position = 1
    for i_step, psd in enumerate(psd_track, 1):
        step = width // (2**i_step)
        r = r[:position] + ("o" if psd else "x") + r[position+1:]
        position = position+step if psd else position-step

    print(r)
    return


def check_psd(inputs):
    """ Checks whether the given matrix is positive semi-definite by performing Cholesky decomposition.

    """
    if isinstance(inputs, np.ndarray) or isinstance(inputs, torch.Tensor):
        M = inputs
        full_matrix_given = True
        to_numpy = isinstance(M, np.ndarray)

    elif isinstance(inputs, tuple) and len(inputs) == 2:
        full_matrix_given = False
        to_numpy = False
        diag_blocks, sub_diag_blocks = inputs

    else:
        raise ValueError(f"Only 1 (matrix) or a list of 2 (iterables) inputs allowed (got {type(inputs)}).")

    if to_numpy:
        try:
            spl.cholesky(M, check_finite=False, overwrite_a=True)
            psd = True
        except spl.LinAlgError:
            psd = False
    else:
        if full_matrix_given:
            try:
                torch.cholesky(M)
                psd = True
            except RuntimeError:
                psd = False
        else:
            try:
                cholesky_block_tridiagonal(diag_blocks, sub_diag_blocks, lower=True, first_diag_block=True)
                psd = True
            except RuntimeError:
                psd = False

    return psd


def largest_eigenvalue(M):
    """ Applies torch.lobpcg if M has more than two rows otherwise uses the explicit formula:

    T = m11 + m22, D = m11*m22 - m12*m21
    sigma_max = 0.5 * (T + sqrt(T^2 - 4*D))
    
    """
    if len(M) == 2:
        T = M[0, 0] + M[1, 1]
        D = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        return 0.5 * (T + torch.sqrt(T*T - 4*D))
    else:
        return torch.lobpcg(M, k=1)[0]


def check_length(*args):
    """ Checks whether the length of the provided arguments are all the same, raises ValueError otherwise.

    """
    length = tuple(map(len, args))
    if not all(l==length[0] for l in length[1:]):
        raise ValueError(f"Length of the inputs {length} don't match.")

    return


def transform_state_dict(state_dict, other, check=False, to_numpy=False, verbose=0):
    """ Transforms the state dict from the DenseNet form into either KWSequential or MIPVerify and vice versa.

    """
    state_dict_is_from_dense_net = any(map(lambda s: s.startswith("layer"), state_dict.keys()))
    state_dict_new = {}

    # from DenseNet to KWSequential
    if state_dict_is_from_dense_net and other=="KWSequential":
        if verbose>0:
            print(f"Convert state dict from DenseNet to {other}.")
        for k, v in state_dict.items():
            if to_numpy:
                v = v.numpy()
            if not k.startswith("layer"):
                split = k.split(".")
                assert split[1] in ("weight", "bias")
                state_dict_new[f"{int(split[0])+1}.{split[1]}"] = v

    # from KWSequential to DenseNet
    elif not state_dict_is_from_dense_net and other=="KWSequential":
        if verbose>0:
            print(f"Convert state dict from {other} to DenseNet.")
        for k, v in state_dict.items():
            if to_numpy:
                v = v.numpy()
            split = k.split(".")
            state_dict_new[f"{int(split[0])-1}.{split[1]}"] = v
            state_dict_new[f"layers.{(int(split[0])-1)//2}.{split[1]}"] = v

    # from DenseNet to MIPVerify
    elif state_dict_is_from_dense_net and other=="MIPVerify":
        if verbose>0:
            print(f"Convert state dict from DenseNet to {other}.")
        for k, v in state_dict.items():
            assert to_numpy
            L = 0
            v = v.cpu().numpy().T
            if k.startswith("layer"):
                split = k.split(".")
                assert split[2] in ("weight", "bias")
                L = max(L, int(split[1])+1)
                state_dict_new[f"fc{int(split[1])+1}/{split[2]}"] = v
        state_dict_new["logits/weight"] = state_dict_new.pop(f"fc{L}/weight")
        state_dict_new["logits/bias"] = state_dict_new.pop(f"fc{L}/bias")

    # from KWSequential to MIPVerify
    elif not state_dict_is_from_dense_net and other=="MIPVerify":
        if verbose>0:
            print(f"Convert state dict from {other} to DenseNet.")
        raise NotImplementedError()

    # from DenseNet to CROWN
    elif state_dict_is_from_dense_net and other=="CROWN":
        if verbose>0:
            print(f"Convert state dict from DenseNet to {other}.")
        for k, v in state_dict.items():
            assert to_numpy
            L = 0
            v = v.cpu().numpy().T
            if k.startswith("layer"):
                split = k.split(".")
                assert split[2] in ("weight", "bias")
                L = max(L, int(split[1])+1)
                state_dict_new[f"{'' if split[2]=='weight' else 'bias_'}U{int(split[1])+1}"] = v

    # invalid inputs
    else:
        raise ValueError("Invalid combination of parameters 'state_dict' and 'other'.")
    
    if check:
        return state_dict_new, state_dict_is_from_dense_net
    else:
        return state_dict_new


def show_available_results(path=None, contains=None, count_only=False):
    """ Prints all files vailable in a given directory (dafault is rdl.PATH_RESULTS).

    """
    path = PATH_RESULTS if path is None else path

    if isinstance(contains, str):
        contains = (contains,)

    n_found = 0
    for p in path.iterdir():
        if contains is not None and all(s in p.name for s in contains):
            if not count_only:
                print(p.name)
            n_found += 1

    return n_found


def drop_last_layer(state_dict):
    """ Given a state dict from a DenseNet objects returns it without the weights/biases corresponding to the last layer.

    """
    if state_dict is None:
        return None

    state_dict_reduced = {}
    L = int(len(state_dict) / 4)

    for k, v in state_dict.items():
        if not k.startswith("layers") and not k.startswith(f"{(L-1)*2}"):
            state_dict_reduced[k] = v
        elif k.startswith("layers") and not k.startswith(f"layers.{L-1}"):
            state_dict_reduced[k] = v

    return state_dict_reduced


def get_qobj_matrix(net,
    alpha=None, alpha0=1.0, to_numpy=False, return_blocks=False, factor=1.0):
    """ Computes the matrix defining the quadratic form in the objective function.

    Parameters:
    -----------
    net : rdl.classifier.DenseNet
        Weights are taken from this net.
    alpha : np.array
        Non-negative multipliers.

    Returns:
    --------
    Q : numpy.array

    """
    nl = net.neurons_list[:-1]
    if alpha is None:
        alpha = \
            [np.full((n, ), 1.0) for n in nl] if to_numpy else \
            [torch.full((n, ), 1.0) for n in nl]
    else:
        alpha = \
            [np.full((net.neurons_list[0], ), alpha0)] + alpha if to_numpy else \
            [torch.full((net.neurons_list[0], ), alpha0, device=alpha[0].device)] + alpha

    if to_numpy:
        # if only the blocks are required
        if return_blocks:
            diag_blocks = (factor * np.diag(alpha_) for alpha_ in alpha)
            sub_diag_blocks = (-0.5 * factor * np.diag(alpha[l]) @ W for # NOTE: 0.5 factor (!)
                l, W in enumerate(net.iter_weights(to_numpy=True, return_generator=True, drop_last=True), 1))
            return diag_blocks, sub_diag_blocks

        # diagonal blocks
        Q = np.diag(np.concatenate(alpha))
        # sub-diagonal blocks
        Q[nl[0]:, :-nl[-1]] +=\
            spl.block_diag(*[-np.diag(alpha[l]) @ W for
            l, W in enumerate(net.iter_weights(to_numpy=True, return_generator=True, drop_last=True), 1)])

    else:
        # if only the blocks are required
        if return_blocks:
            diag_blocks = (factor * torch.diag(alpha_) for alpha_ in alpha)
            sub_diag_blocks = (-0.5 * factor * torch.diag(alpha[l]) @ W for # NOTE: 0.5 factor (!)
                l, W in enumerate(net.iter_weights(to_numpy=False, return_generator=True, drop_last=True), 1))
            return diag_blocks, sub_diag_blocks

        # diagonal blocks
        Q = torch.diag(torch.cat(alpha))
        # sub-diagonal blocks
        Q[nl[0]:, :-nl[-1]] +=\
            torch.block_diag(*[-torch.diag(alpha[l]) @ W for
            l, W in enumerate(net.iter_weights(to_numpy=False, return_generator=True, drop_last=True), 1)])

    # make the matrix symmetric since only "one side" was constructed previously
    Q = 0.5 * factor * (Q + Q.T)

    return Q


def get_qobj_vector(net, x_anchor, alpha, to_numpy=False):
    """ Returns vector B from the objective of the QPRel:

    B = [-2*x_anchor, -alpha_1 * bias_1, ... , -alpha_L-1 * bias_L-1]^T
    
    """
    batched = x_anchor.ndim >= 2
    if batched:
        n_batch = x_anchor.shape[0]

    nl = net.neurons_list[:-1]

    if to_numpy:
        if batched:
            raise NotImplementedError("Not implemented for batched x_anchor and to_numpy=True.")
        if isinstance(x_anchor, torch.Tensor):
            x_anchor = x_anchor.detach().cpu().numpy()

        B = np.empty(sum(nl))
        B[:nl[0]] = -2*x_anchor.reshape(-1)

        for l, (start, n) in enumerate(zip(np.cumsum(nl[:-1]), nl[1:]), 1):
            b = net.layers[l-1].bias.detach().cpu().numpy()
            B[start:start+n] = -alpha[l-1]*b

    else:
        if isinstance(x_anchor, np.ndarray):
            x_anchor = torch.Tensor(x_anchor).to(device=net.device)

        if not batched:
            B = torch.empty(sum(nl), device=net.device)
            B[:nl[0]] = -2*x_anchor.reshape(-1)

            for l, (start, n) in enumerate(zip(np.cumsum(nl[:-1]), nl[1:]), 1):
                b = net.layers[l-1].bias
                B[start:start+n] = -alpha[l-1]*b

        else:
            B = torch.empty((n_batch, sum(nl)), device=net.device)
            B[:, :nl[0]] = -2*x_anchor

            for l, (start, n) in enumerate(zip(np.cumsum(nl[:-1]), nl[1:]), 1):
                layer = net.layers[l-1]
                b = layer.bias if not hasattr(layer, "converted_to_linear") else layer.converted_weight[1]
                B[:, start:start+n] = -alpha[l-1]*b


    return B


def get_cone_matrix(net, additional_blocks, to_numpy=True, to_MixedMatrix=False):
    """ Computes the matrix defining the inequalities from the propagation constraints.

    Given weight matrices W(l) (n(l) x n(l-1)) from layers l=1..L-1 returns the following matrix:

    always
    | -W(1)    E(n(1))                           |
    |          -W(2)   E(n(2))                   |
    |                  ...     ...               |
    |                          -W(L-1) E(n(L-1)) |

    for additional_blocks["ub_input"]
    | -E(n(0)) 0       ...                       |

    for additional_blocks["lb_input"]
    | E(n(0))  0       ...                       |

    for additional_blocks["lb_activation"]
    | 0        E(n(1))                           |
    |          0       E(n(2))                   |
    |                  ...     ...               |
    |                          0       E(n(L-1)) |

    for additional_blocks["label_switch"]
    | ...                      0       -c^T*W(L) |

    where E(n) is a n x n identity matrix.

    Parameters:
    -----------
    net : rdl.classifier.DenseNet
        Weights are taken from this net.

    Returns:
    --------
    M : numpy.array
    b : numpy.array

    """
    nl = net.neurons_list[:-1]
    Index = {
        "main" : []
    }
    for key in additional_blocks:
        Index[key] = []

    def update_index(key, n):
        for key_ in Index:
            Index[key_] += [key==key_] * n

    if to_numpy:
        Wlist = net.iter_weights(bias=True, to_numpy=True)
        M = spl.block_diag(*[-W for W, _ in Wlist[:-1]])
        M = np.concatenate([M, np.zeros((M.shape[0], nl[-1]))], axis=1)
        M[:, nl[0]:] += np.eye(M.shape[0])

        b = np.concatenate([bias for _, bias in Wlist[:-1]])
        update_index("main", len(b))

        if "ub_input" in additional_blocks:
            M_ = np.concatenate([-np.eye(nl[0]), np.zeros((nl[0], M.shape[1]-nl[0]))], axis=1)
            b_ = np.full(len(M_), -additional_blocks["ub_input"])
            M = np.concatenate([M, M_], axis=0)
            b = np.concatenate([b, b_])
            update_index("ub_input", len(b_))

        if "lb_input" in additional_blocks:
            M_ = np.concatenate([np.eye(nl[0]), np.zeros((nl[0], M.shape[1]-nl[0]))], axis=1)
            b_ = np.full(len(M_), additional_blocks["lb_input"])
            M = np.concatenate([M, M_], axis=0)
            b = np.concatenate([b, b_])
            update_index("lb_input", len(b_))

        if "lb_activation" in additional_blocks:
            M_ = np.eye(M.shape[1])[nl[0]:, :]
            b_ = np.full(len(M_), additional_blocks["lb_activation"])
            M = np.concatenate([M, M_], axis=0)
            b = np.concatenate([b, b_])
            update_index("lb_activation", len(b_))

        if "label_switch" in additional_blocks:
            c = additional_blocks["label_switch"]
            M_ = np.concatenate([np.zeros((1, M.shape[1]-nl[-1])), -c.T @ Wlist[-1][0]], axis=1)
            b_ = np.dot(c.T, Wlist[-1][1])
            M = np.concatenate([M, M_], axis=0)
            b = np.concatenate([b, b_])
            update_index("label_switch", 1)

    else:
        Wlist = net.iter_weights(bias=True, to_numpy=False)
        M = torch.block_diag(*[-W for W, _ in Wlist[:-1]])
        M = torch.cat([M, torch.zeros((M.shape[0], nl[-1]), device=net.device)], axis=1)
        M[:, nl[0]:] += torch.eye(M.shape[0], device=net.device)

        b = torch.cat([bias for _, bias in Wlist[:-1]])
        update_index("main", len(b))

        if "ub_input" in additional_blocks:
            M_ = torch.cat([-torch.eye(nl[0], device=net.device), torch.zeros((nl[0], M.shape[1]-nl[0]), device=net.device)], axis=1)
            b_ = torch.full((len(M_), ), float(-additional_blocks["ub_input"]), device=net.device)
            M = torch.cat([M, M_], axis=0)
            b = torch.cat([b, b_])
            update_index("ub_input", len(b_))

        if "lb_input" in additional_blocks:
            M_ = torch.cat([torch.eye(nl[0], device=net.device), torch.zeros((nl[0], M.shape[1]-nl[0]), device=net.device)], axis=1)
            b_ = torch.full((len(M_), ), float(additional_blocks["lb_input"]), device=net.device)
            M = torch.cat([M, M_], axis=0)
            b = torch.cat([b, b_])
            update_index("lb_input", len(b_))

        if "lb_activation" in additional_blocks:
            M_ = torch.eye(M.shape[1], device=net.device)[nl[0]:, :]
            b_ = torch.full((len(M_), ), float(additional_blocks["lb_activation"]), device=net.device)
            M = torch.cat([M, M_], axis=0)
            b = torch.cat([b, b_])
            update_index("lb_activation", len(b_))

        if "label_switch" in additional_blocks:
            c = additional_blocks["label_switch"]
            if c.dim() == 1:
                M_ = torch.cat([torch.zeros((1, M.shape[1]-nl[-1]), device=device), -c.T @ Wlist[-1][0]], axis=1)
                b_ = c.T @ Wlist[-1][1] # vector @ vector
                M = torch.cat([M, M_], axis=0)
            else:
                batch_dims = c.shape[:-1]
                M_ = torch.cat(
                    [
                        torch.zeros((*batch_dims, 1, M.shape[1]-nl[-1]), device=net.device),
                        torch.matmul(-c.unsqueeze(-2), Wlist[-1][0])
                    ],
                    axis=-1
                )
                b_ = torch.matmul(c, Wlist[-1][1]).unsqueeze(-1)
                for n_batch in reversed(batch_dims):
                    if not to_MixedMatrix:
                        M = M.unsqueeze(0).repeat_interleave(n_batch, dim=0)
                    b = b.unsqueeze(0).repeat_interleave(n_batch, dim=0)

                if not to_MixedMatrix:
                    # return one tensor with the unbatched part just copied to match the dimentions of M_
                    M = torch.cat([M, M_], axis=-2)
                else:
                    # return the both parts separately to save memory using MixedMatrix class
                    M = MixedMatrix(M, [None, M_], mode="bottom")
                b = torch.cat([b, b_], axis=-1)
            update_index("label_switch", 1)
            

    return M, b, Index


def count_gpus(verbose=0):
    """ Checks available GPUs using torch.cuda.

    Parameters:
    -----------
    verbose : integer
        Verbosity level.

    Returns:
    --------
    on_gpus : list of integers / False
        If any GPU available then equals [0,1,...,#GPUs-1], otherwise False.
    device : torch.device / None
        First device if any available, otherwise None.

    """
    n_devices = torch.cuda.device_count()
    if n_devices>0:
        on_gpus = list(range(n_devices))
        dvc = torch.device(0) # TODO: do something with the hardcoded device
        if verbose>0:
            print(f"Let's use {n_devices} GPUs")
            print(f"Use {dvc} as the main device\n")
    else:
        on_gpus = False
        dvc = None
        if verbose>0:
            print("No GPUs found\n")

    return on_gpus, dvc


def ccr(y, y_true, logits=False):
    """ Computes the correct classification rate provided predicted and true labels.

    """
    if logits:
        y = y.argmax(dim=1)

    return (y==y_true).float().mean()


def sec2str(s):
    """ Convert a number of seconds to a string of the format 'XXhXXmXX.XXXs'.

    """
    h = int(s / 3600)
    s %= 3600
    m = int(s / 60)
    s %= 60
    ms = s%1 * 1000

    return (h>0)*(f"{h}h") + (m>0)*(f"{m:2d}m") + f"{int(s):02}.{int(ms):03}s"


def standardize(X, *arrays):
    """ Standardize the data.
    
    Computes mean and std for each column of X, these are used to transform columns of
    X itself as well as all columns of other provided arrays.

    Parameters:
    -----------
    X : np.array
        Array that determines standardization parameters.
    arrays : list of np.array
        Other arrays

    Returns:
    --------
    mu : np.array
        Means, array of length X.shape[1].
    sigma : np.array
        Standard deviations, array of length X.shape[1].
    Also returns standardized X and arrays.

    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)

    if len(arrays)==0:
        return mu, sigma, (X-mu)/sigma
    elif len(arrays)==1:
        return mu, sigma, (X-mu)/sigma, (arrays[0]-mu)/sigma
    else:
        return mu, sigma, (X-mu)/sigma, tuple((arr[0]-mu)/sigma for arr in arrays)


def normalize(X):
    """

    """
    mi = X.min(axis=0)
    ma = X.max(axis=0)

    return mi, ma, (X-mi) / (ma-mi)


def extract_architecture_type(s):
    name = None

    for out in ("small", "normal", "large"):
        if s.find(f"net-{out}") > -1:
            name = f"mnist_model_{out}"
    
    for out in ((2, 50), (4, 50), (8, 50)):
        if s.find(f"{out[0]}, {out[1]}") > -1 or s.find(f"{out[0]}_{out[1]}") > -1:
            name = f"mnist_model_{out[0]}_{out[1]}"

    if name is not None:
        logging.info(f"Inferred architecture type: {name}")
        return name
    else:
        raise ValueError("Architecture type could not be inferred from the name.")


def merge_first_two_dimentions(tensor):
    return [None,] if tensor is None else tensor.reshape(np.prod(tensor.shape[:2]), *tensor.shape[2:])