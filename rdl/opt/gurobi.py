# -*- coding: utf-8 -*-

import numpy as np
import gurobipy as grb

from scipy.sparse import spdiags

from rdl.utils import check_length


# TOOLBOX with gurobi expressions

def grb8_inner_product(x, y):
    """ GUROBI 8. Inner product of two indexable objects using grb.quicksum.

    Note:
    -----
        Both x and y might be dictionaries (dict or gurobi.tupledict) therefore
        a direct iteration through them will return the keys and not the values.

    """
    check_length(x, y)
    res = grb.quicksum(x[i]*y[i] for i in range(len(x)))
    return res


def grb8_distance_squared(x, y):
    """ Constructs a gurobi expression representing the squared norm of x-y.

    Parameter:
    ----------
    x : numpy.ndarray
    y : gurobi.tupledict (e.g. from gurobi.Model.addVars())

    """
    if not isinstance(x, np.ndarray):
        raise ValueError("First argument (the anchor point) has to be a numpy.ndarray.")
    #if not isinstance(y, grb.tupledict):
    #    raise ValueError(f"Second argument (point in the input space) has to be a gurobi.tupledict (set of variables), got {type(y)}.")

    check_length(x, y)
    y_vars = y.values() # get the list of variables from a dictionary

    res = grb.QuadExpr()
    res.addTerms(np.ones(len(x)), y_vars, y_vars) # quadratic terms
    res.addTerms(-2*x, y_vars)                    # linear terms
    res += np.sum(x*x)                            # constant term
    return res


def grb8_quadratic_form(x, M=None, y=None):
    """ Constucts a quadratic gurobi expression representing x.T * M * y

    """
    y = x if y is None else y
    M = M if M is not None else np.eye(len(x))
    var1 = x if isinstance(x, list) else x.values()
    var2 = y if isinstance(y, list) else y.values()

    q = grb.QuadExpr()
    q.addTerms(
        list(M.reshape(-1)),
        list(np.stack([var1]*len(var1), axis=0).reshape(-1)),
        list(np.stack([var2]*len(var2), axis=1).reshape(-1)))
    return q


def grb9_quadratic_form(x, M=None):
    """ Constructs a quadratic expression x@M@x for x of the MVar-class.

    """
    if M is None:
        return x.x @ (x.A.T @ x.A) @ x.x + 2 * (x.b @ x.A) @ x.x + x.b @ x.b
    else:
        return x.x @ (x.A.T @ M @ x.A) @ x.x + 2 * (x.b @ M @ x.A) @ x.x + x.b @ M @ x.b


def grb_distance_squared(x, y, gurobi=9):
    """ Constructs a gurobi expression representing the squared norm of x-y.

    Parameter:
    ----------
    x : compatible with the @ operation, first argument (after x.reshape(-1))
    y : compatible with the @ operation, second argument (e.g. gurobi.Model.addMVar)

    """
    if gurobi == 9:
        return grb9_quadratic_form(x.reshape(-1) - y)
    elif gurobi == 8:
        return grb8_distance_squared(x.reshape(-1), y)



def grb_propagation_gap(x, w, alpha, gurobi=9):
    """ Constructs a gurobi expression representing the propagation gap:
            sum(l=1..L-1) alpha(l) * x(l).T ( x(l) - (W(l)x(l-1) + b(l)) )

    """
    if gurobi == 9:
        # unfortunately grb.quicksum does't work for quadratic expressions.
        return np.sum([
            x_l @ (spdiags(alpha_l, 0, len(alpha_l), len(alpha_l)) @ w_l.A) @ w_l.x + (alpha_l * w_l.b) @ x_l for
            alpha_l, x_l, w_l in zip(alpha, x[1:-1], w[1:-1])
        ])
    elif gurobi == 8:
        return grb.quicksum(
            grb.quicksum(
                x_li * alpha_li * w_li for
                x_li, alpha_li, w_li in zip(alpha_l, x_l.values(), w_l)
            ) for
            alpha_l, x_l, w_l in zip(alpha, x[1:-1], w[1:-1])
        )


def grb_propagation_gaps(x, w):
    """ Constructs a list of gurobi expressions representing the
    individual summands of the propagation gap without alpha:

            x(l,i) ( x(l,i) - (W(l,i)x(l-1) + b(l,i)) )

    """
    raise NotImplementedError("Inefficient since requires to compute / store terms for each neuron,")
    out = [x_l @ (w_l.A @ w_l.x + w_l.b) for x_l, w_l in zip(x[1:-1], w[1:-1])] # important to have MVar x_l in the second slot
    return out


def grb_objective(x_anchor, x, w, alpha, gurobi=9):
    """ Constructs a gurobi expression representing the objective function of QPRel.

    """
    obj = {
        "distance_squared" : grb_distance_squared(x_anchor, x[0], gurobi=gurobi),
        "propagation_gap" : grb_propagation_gap(x, w, alpha=alpha, gurobi=gurobi),
        "propagation_gaps" : None}

    return obj