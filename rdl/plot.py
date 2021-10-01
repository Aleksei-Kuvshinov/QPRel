# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt


def plot_gap_hist(x, y, r=None, bins=40, pos=[100, 100, 100]):
    """

    """
    values = (x - y) / y
    q1, q5, q9 = np.quantile(values, (0.1, 0.5, 0.9))
    print(f"quantiles:\n  10% {q1:.3f}\n  50% {q5:.3f}\n  90% {q9:.3f}")

    kwargs = {"linestyle":"--", "linewidth":3, "color":"k"}
    plt.hist(values, bins=bins, range=r, histtype="step", linewidth=7, density=False)

    plt.axvline(q1, **kwargs)
    #plt.text(q1, pos[0], f" {q1:.3f}")
    plt.axvline(q5, **kwargs)
    #plt.text(q5, pos[1], f" {q5:.3f}")
    plt.axvline(q9, **kwargs)
    #plt.text(q9, pos[2], f" {q9:.3f}")

    plt.xlabel("relative difference")
    #plt.ylabel("number of points")
    return


def plot_results(res_list):
    """

    """
    for i, res in enumerate(res_list[1:]):
        x_anchor = res["x_anchor"][:2]

        plt.plot(*x_anchor, "bo")
        plt.plot(*res["x_optimal"][:2], "r*")

        phi = np.linspace(0, 2*np.pi, 100)
        plt.plot(
                x_anchor[0] + np.sqrt(res["objective_value"])*np.sin(phi),
                x_anchor[1] + np.sqrt(res["objective_value"])*np.cos(phi),
                "b-")

    plt.axis("equal")
    return


def plot_verification_ratio(lb=None, ub=None, lb_other=None, ub_other=None, no_plot=False,
    other_name="???", norm_name="???", other_name_ub="???", legend_loc="center right"):
    """

    """
    kwargs={"linewidth":1, "markersize":10}
    e = np.linspace(0, np.max((lb, lb_other)) if ub is None else np.max(ub), num=100)
    qs = (0.1, 0.5, 0.9)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if lb is not None:
        vr_lb = np.array([sum(lb>=ee)/len(lb) for ee in e])
        if not no_plot:
            plt.plot(e, vr_lb, label="QPRel-LB", marker="v", color=colors[0], **kwargs)

        if ub is not None:
            vr_ub = np.array([vr_lb[i] + sum(ub<=ee)/len(lb) for i, ee in enumerate(e)])
            if not no_plot:
                plt.plot(e, vr_ub, label="QPRel-LB and -UB", marker="^", color=colors[1], **kwargs)

    if lb_other is not None:
        vr_lb_other = np.array([sum(lb_other>=ee)/len(lb_other) for ee in e])
        if not no_plot:
            plt.plot(e, vr_lb_other, label=other_name, marker="v", color=colors[2], **kwargs)

        if ub_other is not None:
            vr_ub_other = np.array([vr_lb_other[i] + sum(ub_other<=ee)/len(lb_other) for i, ee in enumerate(e)])
            if not no_plot:
                plt.plot(e, vr_ub_other, label=f"{other_name} and {other_name_ub}", marker="^", color=colors[3], **kwargs)

    lb_QPRel_q = [e[min(sum(vr_lb>=q), len(e)-1)] for q in qs]
    print("lb quants QPRel:\n", lb_QPRel_q)
    lb_Other_q = [e[min(sum(vr_lb_other>=q), len(e)-1)] for q in qs]
    print("lb quants other:\n", [e[min(sum(vr_lb_other>=q), len(e)-1)] for q in qs])
    if ub is not None:
        vr_QPRel_min = min(vr_ub)
        print("Min VR QPRel:", min(vr_ub))
    else:
        vr_QPRel_min = None

    if ub_other is not None:
        vr_Other_min = min(vr_ub_other)
        print("Min VR other:", min(vr_ub_other))
    else:
        vr_Other_min = None

    if not no_plot:
        plt.legend(loc=legend_loc)
        plt.xlabel(f"l{norm_name}-perturbation norm")
        plt.ylabel("verification ratio")
    return {
        "lb_QPRel_q" : lb_QPRel_q, "lb_Other_q" : lb_Other_q, "vr_QPRel_min" : vr_QPRel_min, "vr_Other_min" : vr_Other_min}


def plot_bounds(lb=None, ub=None, lb_other=None, c=None, ub_other=None,
                other_name="???", norm_name="???", other_name_ub="???", legend_loc="upper left",
                skip=1, alpha=0.5):
    """

    """
    lw = 5
    mew = 2
    mfc = "none"
    ms = 10
    M = 2 # markersize multiplier for the legend

    marker = {
        "lb" : "-",
        "ub" : "^",
        "lb_other" : "v",
        "ub_other" : "o"}
    marker = {
        "lb" : ".",
        "ub" : "^",
        "lb_other" : "v",
        "ub_other" : "."}
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    if (lb is None) and (ub is None) and (lb_other is None) and (ub_other is None):
        raise ValueError("nothing to plot")

    I = None

    if lb is not None:
        # plot the points sorted according to the values in lb
        I = np.argsort(lb, axis=None)

    if lb is not None:
        lb = lb[I]
        plt.plot(lb[0], marker["lb"],
            markersize=ms*M, linewidth=lw, label="QPRel-LB", color=colors[0], zorder=5)
        plt.plot(lb, marker["lb"],
            markersize=ms, linewidth=lw, color=colors[0], zorder=5)
        if c is not None:
            plt.plot(c*lb, marker["lb"], linewidth=lw, label=f"lower bound times {c}")

    if ub is not None:
        ub = ub if I is None else ub[I]
        plt.plot(ub[0],
            marker["ub"], alpha=alpha, label="QPRel-UB",
            markersize=ms*M, markeredgewidth=mew, markerfacecolor=mfc, color=colors[1])
        plt.plot(range(len(ub))[::skip], ub[::skip],
            marker["ub"], alpha=alpha,
            markersize=ms, markeredgewidth=mew, markerfacecolor=mfc, color=colors[1])

    if lb_other is not None:
        lb_other = lb_other if I is None else lb_other[I]
        plt.plot(lb_other[0],
            marker["lb_other"], alpha=alpha, label=other_name,
            markersize=ms*M, markeredgewidth=mew, markerfacecolor=mfc, color=colors[2])
        plt.plot(range(len(lb_other))[::skip], lb_other[::skip],
            marker["lb_other"], alpha=alpha,
            markersize=ms, markeredgewidth=mew, markerfacecolor=mfc, color=colors[2])

    if ub_other is not None:
        ub_other = ub_other if I is None else ub_other[I]

        ub_other[0] = 0 if ub_other[0] is None else ub_other[0]
        ub_other[-1] = 0 if ub_other[-1] is None else ub_other[-1]
        plt.plot(ub_other[0],
            marker["ub_other"], alpha=alpha, label=other_name_ub, zorder=6,
            markersize=ms*M, markeredgewidth=mew, color=colors[3])
        plt.plot(range(len(ub_other))[::skip], ub_other[::skip],
            marker["ub_other"], alpha=alpha, zorder=6,
            markersize=ms, markeredgewidth=mew, color=colors[3])

    plt.legend(loc=legend_loc, ncol=1, framealpha=0.5, fancybox=True)
    plt.xlabel("#sample")
    plt.ylabel(f"l{norm_name}-perturbation norm")
    return