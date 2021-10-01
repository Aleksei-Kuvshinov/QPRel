# -*- coding: utf-8 -*-

from pickle import load as pload
from pickle import dump as pdump
import os
from pathlib import Path

from operator import xor
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

import rdl
import rdl.classifier as cl
import rdl.dataset as ds


def read_results_QPRel_pandas(
        filename=None, res_list=None, mask_samples=None, columns=("lb", ),
        n_workers=1, verbose=1):
    """ Converts a list of result dictionaries (output from dbqp.apply_dbqp) into a
    pandas.DataFrame containing the most important results for each sample-target pair.

    """
    # check input integrity
    check = (filename is None, res_list is None)
    if not xor(*check):
        raise ValueError(f"Exactly one of the parameters filename and res_list must be given instead of {'none' if all(check) else 'both'}.")

    if check[1]:
        if isinstance(filename, str):
            path = rdl.PATH_RESULTS / filename
        elif isinstance(filename, Path):
            path = filename
            filename = filename.name
        else:
            raise ValueError(f"Given filename must be either a string or a pathlib.Path, but got an object of type {type(filename)}.")

        with open(path, "rb") as f:
            if verbose > 0:
                print(f"Read the full results from\n{path}")
            if mask_samples is None:
                res_list, mask_samples = pload(f)
            else:
                res_list, _ = pload(f)

    elif check[0] and (mask_samples is None):
        mask_samples = np.array([res is not None for res in res_list])

    # extract the metrics
    n_samples = len(res_list)
    n_targets = len(res_list[0])
    I_samples = np.arange(n_samples, dtype=int)[mask_samples]

    res_df = pd.DataFrame(
        index=pd.MultiIndex.from_product((range(n_samples), range(n_targets)),
                                         names=("sample", "target")),
        columns=columns if not (columns is None) else (
            "lb", "lb_rt", "lb_max"))

    p = Pool(n_workers)

    for col in res_df.columns:
        if verbose > 1:
            print("  ", col)

        inputs = [
            (col, res_list[sample][target])
            for sample in I_samples
            for target in range(n_targets)]
        res_df.loc[pd.IndexSlice[I_samples, :], col] = p.map(extract_results_QPRel, inputs)


    anchors = np.full(n_samples, None)
    anchors[mask_samples] = [
        np.arange(n_targets)[[isinstance(res_ij, dict)
        for res_ij in res_list[i]]][0]
        for i in I_samples]
                         
    targets = np.column_stack((np.full(n_samples, None), np.full(n_samples, None)))
    try:
        targets[mask_samples] = [res_list[i][anchors[i]]["target"]
                for i in I_samples]
    except:
        breakpoint()

    return res_df, anchors, targets


def extract_results_QPRel(args):
    """ Collection of functions to extract various results and metrics from the list
    of dictionaries returned by dbqp.apply_dbqp or io.read_results_QPRel_parallel.

    Parameters:
    -----------
    args : tuple (s, res) with
        s : string
            Encodes what should be computed.
        res : list, dict or NoneType
            Output from dbqp.find_adversarial.

    Returns:
    --------
    out

    """
    s, res = args

    if not isinstance(s, str):
        raise ValueError(f"Given s must be a string, instead got an object of type {type(s)}.")

    if res is None:
        return None

    elif s == "lb":
        if isinstance(res, dict):
            out = 0.0
        elif isinstance(res, list):
            try:
                out = res[1]["certified_radius"]
            except KeyError:
                out = np.sqrt(res[1]["objective_value"])
        return out

    elif s == "lb_rt":
        if isinstance(res, dict):
            out = 0.0
        elif isinstance(res, list):
            out = res[1]["runtime"]
        return out

    elif s == "lb_max":
        if isinstance(res, dict):
            out = 0.0
        elif isinstance(res, list):
            out = np.sqrt(res[1]["objective_value_max"])
        return out

    elif s == "propagation_gap":
        if isinstance(res, dict):
            out = 0.0
        elif isinstance(res, list):
            out = res[-1]["propagation_gap"]
        return out


def extract_results_QPRel_from(path, C, columns=None, norm_name="2", save_to=None):
    if columns is None:
        columns = ("lb", "lb_rt", "propagation_gap")

    res_df, _, t = read_results_QPRel_pandas(filename=path, columns=columns, verbose=-1)

    idx = pd.IndexSlice
    Iidx_lb = list(zip(range(len(res_df)//C), list(zip(*t))[0]))
    Iidx_ub = list(zip(range(len(res_df)//C), list(zip(*t))[1]))

    # metrics from target that resulted in LOWER BOUNDS
    lb = res_df.reindex(index=idx[Iidx_lb], columns=("lb",)).to_numpy(dtype=np.float)
    lb_rt = res_df.reindex(index=idx[Iidx_lb], columns=("lb_rt",)).to_numpy(dtype=np.float)
    propagation_gap = res_df.reindex(index=idx[Iidx_ub], columns=("propagation_gap",)).to_numpy(dtype=np.float)

    out = {
        "lb" : lb,
        "lb_rt" : lb_rt,
        "propagation_gap" : propagation_gap
    }

    if save_to is not None:
        with open(save_to, "wb") as f:
            pdump(out, f)

    return out


def cleanup_results_QPRel_parallel(contains_str="THREAD1_", remove=("[TEST]",), verbose=0):
    """ Applies read_results_QPRel_parallel on everything in rdl.PATH_RESULTS, removes files
    with 'THREAD' and '[TEST]'.

    """
    if isinstance(remove, str):
        remove = (remove, )
        print("'remove' was converted to a tuple:", remove)

    done = False
    while not done:
        done = True

        for filename in os.listdir(rdl.PATH_RESULTS):
            # remove a file if necessary
            if (not (remove is None)) and any(s in filename for s in remove):
                os.remove(rdl.PATH_RESULTS / filename)
                if verbose>0:
                    print(filename, "\nremoved")

            # gather the results from different threads 
            if (contains_str is not None) and (contains_str in filename):
                prefix, suffix = filename.split(contains_str)
                read_results_QPRel_parallel(prefix, (suffix,),
                    save=True, out=False, cleanup=True, verbose=verbose)

                # reset the indicator
                done = False
                break

    return
            

def read_results_QPRel_parallel(path, prefix=None, contains_list=None,
    save=True, out=False, cleanup=False, verbose=0, neurips2019=False, prompt_before_removing=True):
    """ Searches in rdl.PATH_RESULTS for relevant files from single threads and combines the results in a .pkl file.

    Parameters:
    -----------
    prefix : string
        Filename has to start with this string, e.g. 'dbqp_MNIST_l2n100_TRAIN_THREAD' (dafault prefix=None)
    contains_list : list of strings
        Filename has to contain all of these strings, e.g. ("scip-50",) (default contains_list=None)
    save : bool
        Indicates whether the gathered results should be saved in a separate file (default save=True)
    out : bool
        Indicates whether the gathered results should be returned as output (default out=False)
    cleanup : bool
        Indicates whether the found files should be removed (default cleanup=False)
    verbose : int
        Verbosity level (default verbose=0)

    """
    PATH_RESULTS = Path(path)

    output = None
    n_found = 0
    
    cleanup_list = []
    for filename in os.listdir(PATH_RESULTS):
        if (prefix is not None) and (not filename.startswith(prefix)):
            continue #else
        if (contains_list is not None) and not all(s in filename for s in contains_list):
            continue #else

        n_found += 1
        if verbose>=2:
            print("found:", filename)

        # construct the filename to save the whole tuple of results at the end
        if n_found==1 and save and ("THREAD" in filename):
            for part in filename.split("_"):
                if part.startswith("THREAD"):
                    filename_full = filename.replace(part + "_", "")
                    break
            print(f"Save the concatenated results in\n{filename_full}")

        # read one thread's output
        with open(PATH_RESULTS / filename, "rb") as f:
            output_thread = pload(f)

        # first one
        if output is None:
            output = list(output_thread)

        # get the mask_samples vector indicating the samples processed by this thread
        mask_samples_thread = output_thread[1]
        I_thread = np.arange(len(mask_samples_thread))[mask_samples_thread]

        for j, value_thread in enumerate(output_thread):
            if isinstance(value_thread, np.ndarray):
                output[j][mask_samples_thread] = value_thread[mask_samples_thread]
            else:
                for i in I_thread:
                    output[j][i] = value_thread[i]

        # delete the current part
        if cleanup:
            cleanup_list.append(filename)

    if cleanup and n_found>0:
        print(f"WARNING: {len(cleanup_list)} files will be deleted" +
             (", proceed? [y/n]" if not prompt_before_removing else ""))
        proceed_to_removing = True if not prompt_before_removing else (input()=="y")
        if proceed_to_removing:
            for filename in cleanup_list:
                os.remove(PATH_RESULTS / filename)

    print(f"\n{n_found} suitable parts found")

    if save:
        if output is not None:
            print(f"Save the concatenated results in\n{filename_full}\n")
            with open(PATH_RESULTS / filename_full, "wb") as f:
                pdump(tuple(output), f)
        else:
            print("Nothing to save\n")

    return tuple(output) if out else None


def read_results_CROWN(dir_name, prefix, N):
    """ Goes through files {prefix}* in rdl.PATH_RESULTS/dir_name and reads in their content.

    """
    PATH_RESULTS = rdl.PATH_RESULTS / dir_name

    dist = np.empty(N)
    dist.fill(None)
    runtime = np.empty(N)
    runtime.fill(None)

    I = []
    for filename in os.listdir(PATH_RESULTS):
        if filename.startswith(prefix):
            i = int(filename[len(prefix):])
            I.append(i) 
            with open(PATH_RESULTS / filename, "rb") as f:
                d = pload(f)
            # d.keys() are "predict_label", "target_label", "robustness_gx", "time"
            dist[i] = d["robustness_gx"]
            runtime[i] = d["time"]

    return dist, runtime, I


def save_net(net, path, model_name, input_size):
    """ Write the layout and the parameters of a NN into a file.

    See:
    ----
    https://pytorch.org/tutorials/beginner/saving_loading_models.html

    """
    torch.save({
        "model_name" : model_name,
        "state_dict" : net.state_dict(),
        "input_size" : input_size},
        path)

    return


def load_net(path, device=None, architecture_type=None, **kwargs):
    """ Loads a model from a file saved with save_net.

    """
    if device is None:
        device = torch.device("cpu")

    d = torch.load(path, map_location=device)
    model_name, state_dict, input_size = d["model_name"], d["state_dict"], d["input_size"]
    if architecture_type is None:
        architecture_type = model_name

    net = getattr(cl, architecture_type)(**kwargs)
    net = cl.SequentialNet(
        list(net),
        input_size=input_size, device=device)
    net.load_state_dict(state_dict)

    return net


def get_other_data_from(PATH_DATA=None,
    classes=2, reduce_dataset=1, batch_size=8192, device=None, standartize=False, normalize=False):
    """ Tries to load a dictionary from path using pickle module.

    """
    if isinstance(PATH_DATA, str):
        PATH_DATA = Path(PATH_DATA)
    elif PATH_DATA is None:
        print("No PATH_DATA provided, use manual input:")
        PATH_DATA = str(input())
        return get_other_data_from(PATH_DATA=PATH_DATA,
            classes=classes, reduce_dataset=reduce_dataset, batch_size=batch_size, device=device)
    elif not isinstance(PATH_DATA, Path):
        raise ValueError(f"PATH_DATA has to be either a string or a pathlib.Path, given object/type:\n{PATH_DATA}\n{type(PATH_DATA)}")

    with open(PATH_DATA, "rb") as f:
        X, y = pload(f)

    if standartize:
        _, _, X = rdl.utils.standardize(X)
    if normalize:
        _, _, X = rdl.utils.normalize(X)

    # convert data to pytorch tensors
    X, y = torch.Tensor(X), torch.LongTensor(y)
    if device is not None:
        X, y = X.to(device), y.to(device)

    return ds.Dataset_(X, y, classes=classes, reduce_dataset=reduce_dataset, batch_size=batch_size)


def get_uci_data(name, normalize=True, batch_size=None):
    path = rdl.PATH_DATA / "uci" / name
    try:
        with open(path / "data", "rb") as f:
            data = pload(f)
        with open(path / "target", "rb") as f:
            target = pload(f)
    except:
        import py_uci as uci

        dataset = uci.get(name)
        data, target = dataset.data, dataset.target
        with open(path / "data", "wb") as f:
            pdump(dataset.data, f)
        with open(path / "target", "wb") as f:
            pdump(dataset.target, f)

    if normalize:
        _, _, data = rdl.utils.normalize(data)

    return rdl.dataset.Dataset_(data, target, name=name, batch_size=batch_size if batch_size else len(data))