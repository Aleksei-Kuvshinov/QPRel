# -*- coding: utf-8 -*-

import numpy as np
import gurobipy as grb
from scipy.sparse import spdiags

import torch

from pickle import dump as pdump
from pickle import load as pload

from time import time
from multiprocessing import Pool
import os

import rdl
import rdl.utils as utils
import rdl.opt.gurobi as grb_toolbox
from rdl.utils import inf
from rdl.io import cleanup_results_QPRel_parallel as cleanup




class DBQP():
    """ Class implementing a QP instance for finding the distance to the decision boundary.

    Attributes:
    -----------
    net : classifier.DenseNet
        Underlying NN, see classifier.py for details.
    x_anchor : torch.Tensor
        Distance to the decision boundary is computed from this point.
    y_anchor : int
        Class label of x_anchor.
    y_target : int
        Class label that an adversarial example should have.
    alpha : numpy.ndarray
        Positive multiplication parameter for the propagation penalty (best we could find).
    alpha_max : numpy.ndarray
        Positive multiplication parameter for the propagation penalty (one that certainly results in a non-convex problem).
    box_bounds_input : 2-tuple of floats
        Lower and upper bound on the input (each feature independently, so called box constraints).
    model : gurobipy.Model
        Object containing the QP implemented in gurobipy.
    x : list of gurobi variable containers
        List, each element x[l] is a container of gurobi variables from the corresponding layer.
    w : list of dictionaries
        w[l][i] is a gurobi linear expression representing the propagation slackness in the i's node of the l's layer: x(l,i) - (W(l,i)x(l-1) + b(l)).
    obj : dict
        Dictionary containing gurobi objects representing parts of the quadratic objective function.

    """


    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------- INIT ----------------------------------------- #
    # ------------------------------------------------------------------------------------------ #

    def __init__(self, net, x_anchor, y_anchor, y_target, alpha=None,
                 box_bounds_input=(0.0, 1.0),
                 norm_bounds_input_init={"i":None, "2":None},
                 box_bounds_activation={"i":None, "2":None},
                 name="dbqp", n_threads=0, norm_type="2", mu_type=None,
                 gurobi=8, verbose=1):
        """ Builds the DBQP object containing a gurobi model for the QPRel problem.

        Parameters:
        -----------
        net : see DBQP
        x_anchor : see DBQP
        y_anchor : see DBQP
        y_target : see DBQP
        alpha : see DBQP and DBQP.set_alpha_opt()
        box_bounds_input : tuple
            Lower and upper bound on the adversarial example in the input space, e.g. (0, 1) for images.
        name : string
            Name for the gurobi model.
        n_threads : int
            Number of threads allowed to be used by gurobi.
        verbose : int
            Verbosity level.

        """
        if verbose >= 2:
            t_init = time()
            print("[DBQP.__init__]: START")

        self.misc = {
            "bnd_u" : 1.0,
        }

        implemented_norms = ("2", "i")
        if not norm_type in implemented_norms:
            raise ValueError(f"WARNING: provided norm {norm_type} is not supported, implemented norms are {implemented_norms}")

        self.net = net
        if x_anchor is None:
            x_anchor = np.zeros([1] + list(net.input_size))
        self.x_anchor = x_anchor

        self.gurobi = gurobi
        self.y_target = y_target
        self.alpha = alpha
        self.box_bounds_input = box_bounds_input
        self.norm_bounds_input = None
        self.norm_bounds_input_init = norm_bounds_input_init # bounds that should hold always

        self.norm_type = norm_type
        if norm_type == "2":
            mu = 1.0
        elif norm_type == "i":
            if mu_type == "normal":
                mu = 1.0 / ((box_bounds_input[1]-box_bounds_input[0]) * net.neurons_list[0])
            elif mu_type == "improved":
                mu = 1.0 / net.neurons_list[0]
            elif mu_type == "old":
                mu = 1.0 / (2.0*net.neurons_list[0])
            else:
                raise ValueError(f"Got an invalid mu_type {mu_type}.")
        self.mu = mu
        self.mu_type = mu_type

        self.model = grb.Model(name=name)
        self.model.setParam("Threads", n_threads)

        self.additional_constraints = {
            "box_bounds_activation_lower" : [],
            "box_bounds_activation_upper" : [],
            "relu_relax_uncertain" : [],
            "relu_relax_deactivate" : [],
            "relu_relax_activate" : [],
            "quad_constr" : []
        }

        # VARS
        ns = net.neurons_list
        ls = len(ns)

        x = []
        w = [None,]
        for l in range(ls):
            nsl = int(ns[l])
            # no/given bounds on x(0)
            if l == 0:
                if box_bounds_input is not None:
                    lb_input = -grb.GRB.INFINITY if box_bounds_input[0] is None else box_bounds_input[0]
                    ub_input = grb.GRB.INFINITY if box_bounds_input[1] is None else box_bounds_input[1]
                else:
                    lb_input = -grb.GRB.INFINITY
                    ub_input = grb.GRB.INFINITY

                if gurobi == 9:
                    x.append(self.model.addMVar(
                        nsl, lb=lb_input, ub=ub_input, name=f"x{l}"
                    ))
                elif gurobi == 8:
                    x.append(self.model.addVars(
                        nsl, lb=lb_input, ub=ub_input, name=f"x{l}"
                    ))
                    

                # l-infinity case: additional variable and quad-conic constraints
                if norm_type == "i" and mu_type == "old":
                    if gurobi == 9:
                        self.m = self.model.addMVar(1, lb=0.0, name=f"m")
                    elif gurobi == 8:
                        self.m = self.model.addVar(lb=0.0, name=f"m")

                    self.add_quad_constr(x0=x[0])

            # no lower bound on x(L)
            elif l == ls-1:
                #x.append(self.model.addMVar(
                #    nsl, lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, name=f"x{l}"))
                if gurobi == 9:
                    x.append(None)
                elif gurobi == 8:
                    x.append([None] * nsl)

                w.append(None)

            # non-negativity constraint for other x(l)
            else:
                if gurobi == 9:
                    x.append(self.model.addMVar(
                        nsl, lb=0.0, ub=grb.GRB.INFINITY, name=f"x{l}"))
                elif gurobi == 8:
                    x.append(self.model.addVars(
                        nsl, lb=0.0, ub=grb.GRB.INFINITY, name=f"x{l}"))
                w.append([None] * nsl)

        self.x = x

        # CONS

        ## definition of w
        for l, (weight, bias) in enumerate(self.net.iter_weights(bias=True, return_generator=True, to_numpy=True), 1):
            nsl = ns[l]
            if l == ls-1:
                if gurobi == 9:
                    x[l] = weight @ x[l-1] + bias
                elif gurobi == 8:
                    for i in range(nsl):
                        x[l][i] = grb_toolbox.grb8_inner_product(weight[i], x[l-1]) + bias[i]
            else:
                if gurobi == 9:
                    w[l] = x[l] - (weight @ x[l-1] + bias)
                    self.model.addConstr(
                        w[l] >= 0,
                        name=f"propagation({l})")
                elif gurobi == 8:
                    for i in range(nsl):
                        w[l][i] = x[l][i] - (grb_toolbox.grb8_inner_product(weight[i], x[l-1]) + bias[i])
                        self.model.addConstr(
                            w[l][i] >= 0,
                            name=f"propagation({l},{i})")

        self.w = w

        ## label switch
        self.set_labels(y_anchor, y_target, remove_old_label_switch_constr=False)

        ## intermediate activations bounds
        self.set_intermediate_activation_bounds(
            norm_bounds_input_init, update_mu=False, verbose=verbose)

        # OBJF & ALPHA
        if alpha is None:
            self.set_alpha_opt(verbose=verbose, update_objective=False)

        self.update_objective()

        # shut off any logging from gurobi
        self.model.setParam("OutputFlag", 0)

        if verbose >= 2:
            print("[DBQP.__init__]: DONE", time() - t_init, "\n")
        return


    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------- CHCK ----------------------------------------- #
    # ------------------------------------------------------------------------------------------ #

    def check_psd(self):
        """

        """
        return utils.check_psd(self.get_qobj_matrix(from_net=True))


    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------- SET_ ----------------------------------------- #
    # ------------------------------------------------------------------------------------------ #

    def update_objective(self, part="full"):
        """ Updates the objective function according to the current attributes.

        Parameters:
        -----------
        part : string ('dist', 'comp', 'mu' or 'full')
            Determines what part of the objective should be updated (default part='full').

        """
        gurobi = self.gurobi

        if part is "full":
            print("update_objective: FULL")
            self.obj = grb_toolbox.grb_objective(self.x_anchor, self.x, self.w, self.alpha, gurobi=gurobi)
        elif not hasattr(self, "obj"):
            raise ValueError(f"Attribute 'obj' wasn't defined yet, hence only part='full' allowed (given {part}).")

        elif part is "dist":
            print("update_objective: DIST")
            self.obj["distance_squared"] = grb_toolbox.grb_distance_squared(
                self.x_anchor, self.x[0], gurobi=gurobi)
        elif part is "comp":
            print("update_objective: COMP")
            self.obj["propagation_gap"] = grb_toolbox.grb_propagation_gap(
                self.x, self.w, alpha=self.alpha, gurobi=gurobi)
        elif part is "mu":
            pass
        else:
            raise ValueError(f"Only valid parameter values are 'dist', 'comp' and 'full' (got {part}).")

        if self.norm_type == "i" and self.mu_type == "old":
            OBJ = self.mu*self.obj["distance_squared"] + self.obj["propagation_gap"] + \
                self.m*(1.0 - self.mu*self.net.neurons_list[0])
            self.model.setObjective(OBJ)
        else:
            OBJ = self.mu*self.obj["distance_squared"] + self.obj["propagation_gap"]
            self.model.setObjective(OBJ)

        # NOTE: BUGFIX, propagation_gap object gets changed by
        #       setObjective() (!!!), so we manually reset it.
        self.obj["propagation_gap"] = grb_toolbox.grb_propagation_gap(self.x, self.w, alpha=self.alpha, gurobi=gurobi)
        return


    def set_labels(self, y_anchor, y_target, verbose=0, remove_old_label_switch_constr=True):
        """ Resets the source/target values by changing the coefficients of the constraint 'label swith'.

        Parameters:
        -----------
        y_anchor : integer
            New source label.
        y_target : integer
            New target label.
        verbose : int
            Verbosity level.

        """
        self.model.update()
        if verbose > 1:
            print(f"anchor {self.y_anchor} -> {y_anchor}, target {self.y_target} -> {y_target}")

        if remove_old_label_switch_constr:
            constr = self.model.getConstrByName("label_switch")
            if constr is not None:
                self.model.remove(constr)
            else:
                raise Warning("[set_labels]: label_switch constraint wasn't found!")

        self.y_anchor = int(y_anchor)
        self.y_target = int(y_target)

        if self.gurobi == 9:
            c = np.zeros(self.net.n_classes)
            c[y_anchor], c[y_target] = 1.0, -1.0
            self.model.addConstr(
                c @ self.x[-1].A @ self.x[-1].x + c @ self.x[-1].b <= 0.0,
                #self.x[-1][y_anchor] - self.x[-1][y_target] <= 0.0,
                name="label_switch")
        elif self.gurobi == 8:
            self.model.addConstr(
                self.x[-1][y_anchor] - self.x[-1][y_target] <= 0.0,
                name="label_switch")

        return


    def set_mu(self, norm_bound_input, verbose=0):
        """ Resets the mu factor for the new l_inf bound on the input perturbation.

        NOTE: should be used in the l_inf case only!
        """
        if not self.norm_type == "i":
            raise ValueError(
                f"set_mu called with self.norm_type being {self.norm_type} (only 'i' allowed).")

        norm_i = (self.box_bounds_input[1]-self.box_bounds_input[0]) \
            if norm_bound_input is None else norm_bound_input

        mu_old = self.mu
        self.mu = 1.0 / (self.net.neurons_list[0] * norm_i)
        if self.mu >= mu_old:
            self.set_alpha_opt(
                alpha_min=self.alpha, alpha_max=None, update_objective=True, verbose=0)
            #pass # we are still guarantied to have a PSD matrix
        else:
            self.set_alpha_opt(
                alpha_min=None, alpha_max=self.alpha, update_objective=True, verbose=0)

        return


    def set_intermediate_activation_bounds(self, norm_bounds_input, update_mu=True, verbose=0):
        """ Reset the additional linear constraints from bounds propagation

        """
        gurobi = self.gurobi

        self.model.update()
        if self.norm_type == "i" and update_mu and self.mu_type == "improved":
            self.set_mu(norm_bound_input=norm_bounds_input["i"], verbose=verbose)

        # remove current constraints
        # NOTE: there is no way to efficiently remove a group of constraints
        #       hence we go through the whole list of the constraints here:
        #       "box_bounds_activation_lower", "box_bounds_activation_upper",
        #       "relu_relax_uncertain", "relu_relax_activate", "relu_relax_deactivate"
        for cname in self.additional_constraints.keys():
            for constr in self.additional_constraints[cname]:
                self.model.remove(constr)
            self.additional_constraints[cname] = []

        # reset activation bounds
        self.norm_bounds_input = norm_bounds_input
        box_bounds_activation = {"i":None, "2":None}
        for norm_type_, epsilon in norm_bounds_input.items():
            if epsilon is not None:
                if verbose >= 2:
                    print(f"    [set_intermediate_activation_bounds]: got an upper bound on input's {norm_type_}-norm: {epsilon:.6}")
                box_bounds_activation[norm_type_] = self.net.get_activation_bounds(
                    torch.Tensor(self.x_anchor), epsilon, norm_type_)

        ns = self.net.neurons_list
        ls = len(ns)

        available_norm_types = [
            norm_type_
            for norm_type_, box_bounds_activation_ in box_bounds_activation.items()
            if box_bounds_activation_ is not None
        ]

        if any(available_norm_types):
            ## bounds on activation
            if verbose >= 3:
                print("    [set_intermediate_activation_bounds]: reset intermediate activation bounds")
            for l in range(1, ls-1):
                nsl = ns[l]

                if gurobi == 9:
                    bba_lower = np.max(
                        [np.array(box_bounds_activation[norm_type_][l-1])[:, 0] for norm_type_ in available_norm_types],
                        axis=0)
                    bba_upper = np.min(
                        [np.array(box_bounds_activation[norm_type_][l-1])[:, 1] for norm_type_ in available_norm_types],
                        axis=0)

                    assert (bba_upper >= bba_lower).all()

                    ### upper & lower
                    cname = "box_bounds_activation_lower"
                    self.additional_constraints[cname] += \
                        self.model.addConstr(
                            self.x[l] - self.w[l] >= bba_lower, name=f"{cname}({l})")
                    cname = "box_bounds_activation_upper"
                    self.additional_constraints[cname] += \
                        self.model.addConstr(
                            self.x[l] - self.w[l] <= bba_upper, name=f"{cname}({l})")

                    ### additional linear
                    I = (bba_lower < 0.0) & (bba_upper > 0.0)
                    In = np.sum(I)
                    Ia = bba_lower >= 0.0
                    Id = bba_upper <= 0.0
                    if verbose >= 3:
                        print(f"    [set_intermediate_acitvation_bounds]: uncertain/active/inactive neurons in layer ({l})", In, np.sum(Ia), np.sum(Id))

                    #### uncertain neurons
                    if I.any():
                        cname = "relu_relax_uncertain"
                        self.additional_constraints[cname] += \
                            self.model.addConstr(
                                spdiags(bba_upper[I], 0, In, In) @ self.w[l].A[I] @ self.w[l].x + bba_upper[I] * self.w[l].b[I] <=
                                spdiags(bba_lower[I], 0, In, In) @ self.x[l][I] - bba_lower[I] * bba_upper[I],
                                name=f"{cname}({l})")

                    #### activated neurons
                    if Ia.any():
                        cname = "relu_relax_activate"
                        self.additional_constraints[cname] += \
                            self.model.addConstr(
                                self.w[l].A[Ia] @ self.w[l].x + self.w[l].b[Ia] == 0,
                                name=f"{cname}({l})")

                    #### deactivated neurons
                    if Id.any():
                        cname = "relu_relax_deactivate"
                        self.additional_constraints[cname] += \
                            self.model.addConstr(
                                self.x[l][Id] == 0,
                                name=f"{cname}({l})")

                elif gurobi == 8:
                    for i in range(nsl):
                        #if box_bounds_activation[self.norm_type] is not None:
                        available_norm_types = [
                            norm_type_
                            for norm_type_, box_bounds_activation_ in box_bounds_activation.items()
                            if box_bounds_activation_ is not None]

                        bba_lower = None if len(available_norm_types) == 0 else\
                        max([
                            box_bounds_activation[norm_type_][l-1][i][0]
                            for norm_type_ in available_norm_types])
                        bba_upper = None if len(available_norm_types) == 0 else\
                        min([
                            box_bounds_activation[norm_type_][l-1][i][1]
                            for norm_type_ in available_norm_types])

                        # upper & lower
                        cname = f"box_bounds_activation_lower({l,i})"
                        c = self.model.getConstrByName(cname)
                        if c is not None:
                            self.model.remove(c)
                        if len(available_norm_types) != 0:
                            self.model.addConstr(
                                self.x[l][i] - self.w[l][i] >= bba_lower, name=cname)

                        cname = f"box_bounds_activation_upper({l,i})"
                        c = self.model.getConstrByName(cname)
                        if c is not None:
                            self.model.remove(c)
                        if len(available_norm_types) != 0:
                            self.model.addConstr(
                                self.x[l][i] - self.w[l][i] <= bba_upper, name=cname)

                        # additional linear
                        cname = f"relu_relax({l,i})"
                        c = self.model.getConstrByName(cname)
                        if c is not None:
                            self.model.remove(c)

                        if len(available_norm_types) != 0:
                            if bba_lower < 0 and bba_upper > 0:
                                self.model.addConstr(
                                    self.x[l][i]*(bba_lower - bba_upper) >= bba_upper*
                                        (bba_lower - (self.x[l][i]-self.w[l][i])),
                                    name=cname)
                            elif bba_upper <= 0:
                                self.model.addConstr(self.x[l][i] == 0, name=cname)
                            elif bba_lower >= 0:
                                self.model.addConstr(self.w[l][i] == 0, name=cname)


            if verbose >= 3:
                print("    [set_intermediate_activation_bounds]: DONE,")

        return


    def set_anchor(self, x_anchor, verbose=0):
        """ Resets the objective function with a new anchor point.

        NOTE: also updates the objective.

        Parameters:
        -----------
        x_anchor : torch.Tensor
            New anchor point.
        verbose : int
            Verbosity level.

        """
        self.x_anchor = x_anchor
        self.update_objective(part="dist")

        # for norm "i" we have to update constraints as well
        if self.norm_type == "i" and self.mu_type == "old":
            self.model.remove(self.model.getQConstrs())
            self.additional_constraints["quad_constr"] = []
            self.add_quad_constr()

        if verbose >= 2:
            print("  [set_anchor]: anchor point reset")
        return


    def set_alpha(self, alpha, update_objective=False):
        """ Resets the value of alpha by resetting the objective.

        Parameters:
        -----------
        alpha : list of floats
            New value for alpha.

        """
        self.alpha = alpha
        if update_objective:
            # NOTE: alpha occures in the propagation gap within the objective
            #       function only. Hence, there is no need to update the constraints.
            self.update_objective(part="comp")

        return


    def set_alpha_opt(self, update_objective=False, **kwargs):
        """ Performs a binary search between alpha_safe and alpha_max.

        See utils.get_alpha_opt()

        """
        alpha_min = kwargs.pop("alpha_min", self.alpha)
        alpha0 = kwargs.pop("alpha0", self.mu)

        alpha_opt = utils.get_alpha_opt(self.net, alpha_min=alpha_min, alpha0=alpha0, **kwargs)
        self.set_alpha(alpha_opt, update_objective=update_objective)

        return


    def add_quad_constr(self, x0=None):
        """ Adds the following quadratic constraints to the model.

        (x[0] - x_anchor) ^ 2 <= m

        NOTE: should be used for norm_type='i' and mu_type='old'.

        """
        if not (self.norm_type == "i" and self.mu_type == "old"):
            raise ValueError(
                f"add_quad_constr is used only for norm_type = 'i' ({self.norm_type}) and mu_type = 'old' ({self.mu_type}).")

        if x0 is None:
            x0 = self.x[0]

        for i in range(self.net.neurons_list[0]):
            x_anchor_i = self.x_anchor.reshape(-1)[i]
            self.additional_constraints["quad_constr"].append(
                self.model.addConstr(
                    x0[i] @ x0[i] - 2 * x_anchor_i * x0[i] + x_anchor_i * x_anchor_i <= self.m,
                    name=f"quad_constr({i})"
                    )
            )


    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------- GET_ ----------------------------------------- #
    # ------------------------------------------------------------------------------------------ #

    def get_qobj_matrix(self, from_net=True, from_gurobi=False):
        """ Computes the matrix defining the quadratic form in the objective function.

        Returns:
        --------
        Q : numpy.array

        Note:
        -----
        from_gurobi=True mode is for testing/debugging purposes only.

        """
        if from_gurobi == from_net:
            raise ValueError(f"from_gurobi ({from_gurobi}) and from_net ({from_net}) have to be different.")

        nl = self.net.neurons_list[:-1]

        if from_net:
            Q = utils.get_qobj_matrix(self.net, self.alpha, alpha0=self.mu)

        if from_gurobi:
            raise DeprecationWarning("This might not work after migrating to gurobi9.")
            n = sum(nl)
            Q = np.zeros((n, n))
            names = [var.varName for var in self.model.getVars()]

            obj = self.model.getObjective()
            for i in range(obj.size()):
                var1 = obj.getVar1(i)
                var2 = obj.getVar2(i)
                Q[names.index(var1.varName), names.index(var2.varName)] = obj.getCoeff(i)

        return Q


    def get_var_value(self, var):
        """ Returns the current value of a variable.

        """
        gurobi = self.gurobi

        if gurobi == 9:
            return var.X

        elif gurobi == 8:
            if isinstance(var, grb.Var):
                return var
            elif isinstance(var, dict):
                return np.array([x_.X for x_ in var.values()])
            else:
                raise ValueError(f"Got 'var' of an unsupported type ({type(var)}).")


    def get_expr_value(self, expr):
        """ Returns the current value of an expression.

        """
        if isinstance(expr, grb.GenExpr):
            return expr.getValue()
        elif isinstance(expr, list):
            return np.array([expr_.getValue() for expr_ in expr])
        else:
            raise ValueError(f"Got 'expr' of an unsupported type ({type(expr)}).")


    def get_xopt(self, full=False):
        """ Extracts optimal set of x variables for the first layer from an optimized model.

        """
        if not full:
            return self.get_var_value(self.x[0])
        else:
            return np.array([self.get_var_value(x_l) for x_l in self.x[:-1]])


    def get_propagation_gap(self, alpha=None, x_full=None):
        """ Computes the value of the propagation gap for the current values of x and given values of alpha.

        """
        alpha = alpha if alpha is not None else self.alpha

        if x_full is None:
            res1 = np.sum((alpha_l * self.get_var_value(x_l)) @ self.get_expr_value(w_l)for alpha_l, x_l, w_l in zip(alpha, self.x[1:-1], self.w[1:-1]))
        else:
            res1 = np.sum(
                (alpha[l-1] * x_full[l]) @ (x_full[l] - (weight @ x_full[l-1] + bias))
                for l, (weight, bias) in enumerate(self.net.iter_weights(bias=True, return_generator=True, drop_last=True, to_numpy=True), 1))

        return res1


    def get_distance(self, x, y, norm=None, squared=False):
        """ Returns the distance between the input points w.r.t. to self.norm.

        """
        if norm is None:
            norm = self.norm_type

        x = x.reshape(-1)
        y = y.reshape(-1)
        d = x - y

        if norm == "2":
            return np.sum(d**2) if squared else np.sqrt(np.sum(d**2))
        elif norm == "i":
            return np.max(np.abs(d)) if not squared else np.max(np.abs(d))**2
        else:
            raise ValueError(f"Given norm ({norm}) is not supported.")


    def get_objective_value(self, x_full=None):
        """ Returns the current objective function value.

        """
        x0_opt = self.get_xopt() if x_full is None else x_full[0]

        distance_squared_2 = self.get_distance(x0_opt, self.x_anchor, norm="2", squared=True)
        propagation_gap = self.get_propagation_gap(x_full=x_full)

        # case: 2-norm
        if self.norm_type == "2":
            distance_squared = distance_squared_2
        # case: i-norm
        elif self.norm_type == "i":
            distance_squared = self.get_distance(x0_opt, self.x_anchor, norm="i", squared=True)

        objective_value = self.mu * distance_squared_2 + propagation_gap
        certified_radius = \
            objective_value if (self.norm_type == "i" and self.mu_type == "improved") else \
            np.sqrt(objective_value)

        return {
            "objective_value" : objective_value,
            "distance_squared" : distance_squared,
            "propagation_gap" : propagation_gap,
            "certified_radius" : certified_radius
        }


    def get_solution(self):
        output = {
            "x0_opt" : self.get_xopt(),
            "model_status" : self.model.status,
            **self.get_objective_value()
        }

        return output

    # ------------------------------------------------------------------------------------------ #
    # ------------------------------------------- MAIN ----------------------------------------- #
    # ------------------------------------------------------------------------------------------ #

    def binary_search_for_TESTbnd(self, bnd_l, bnd_u,
                                  max_iter=10, no_binary_search=False, verbose=-1, threshold=1e-8):
        """ Find the lowest bound such that the problem remains feasible.

        """

        if verbose >= 2:
            print("  [binary_search_for_TESTbnd]: START")
        bnd_u_found = False
        i_step = 0
        while i_step < max_iter and (bnd_u - bnd_l > threshold or not bnd_u_found):

            if bnd_u_found:
                i_step += 1
                bnd = bnd_l + 0.5 * (bnd_u - bnd_l)
            else:
                bnd = bnd_u

            self.set_intermediate_activation_bounds(
                norm_bounds_input={
                    "i":bnd if self.norm_type == "i" else self.norm_bounds_input_init["i"],
                    "2":bnd if self.norm_type == "2" else self.norm_bounds_input_init["2"]
                    },
                verbose=verbose)

            self.model.optimize()

            if not bnd_u_found:
                if self.model.status == 2 and \
                self.get_objective_value()["certified_radius"] <= bnd: # found one
                    bnd_u_found = True
                    print("  [binary_search_for_TESTbnd]: found an upper bound, start binary search.")
                    # get optimal solution and objective function value
                    output = self.get_solution()
                    if no_binary_search:
                        break
                else:
                    if self.model.status == 2:
                        print(f"  [binary_search_for_TESTbnd]: certified radius (l{self.norm_type})",
                              self.get_objective_value()["certified_radius"])
                    bnd_l = bnd_u
                    bnd_u *= 2

            else:
                if self.model.status == 2 and \
                self.get_objective_value()["certified_radius"] <= bnd:
                    bnd_u = bnd
                    # get optimal solution and objective function value
                    output = self.get_solution()
                else: # might be 4, 12, 13
                    bnd_l = bnd

        if verbose >= 2:
            print("  [binary_search_for_TESTbnd]: DONE")
        return {"bnd_u" : bnd_u, **output}


    def find_adversarial(self, eps_compl=1e-8, eps_alpha=1e-8,
                         alpha_step=1e-5, n_threads=1, delta=1e-2,
                         relative_bound_improvement_threshold=1e-3,
                         verbose=0, TESTeach=1, n_gamma=10):
        """ Solves the QP relaxation.

        Returns:
        --------
        res : list of dicts
            Each element is a dict containing information about the corresponding iteration including e.g. solution of QPRel, note that the first element includes only the setup information.

        """
        y_predicted = self.net.labels(self.x_anchor).item()
        res = [{
            "x_anchor" : self.x_anchor,
            "x_optimal" : None,
            "distance_squared" : inf,
            "objective_value" : inf,
            "objective_value_max" : inf,
            "propagation_gap" : inf,
            "alpha" : self.alpha,
            "status" : self.model.status,
            "runtime" : 0.0,
            "y_anchor" : self.y_anchor,
            "y_target" : self.y_target,
            "y_predicted" : y_predicted,
            "eps_compl" : eps_compl,
            }]

        t = time()
        if verbose > 2:
            print("\n[find_adversarial]: Optimization start")

        # Find the tightest input bounds and construct the additional constraints
        if TESTeach > 0:
            if verbose >= 2:
                print("  [find_adversarial]: find the best upper bound for bound propagation")
            output = self.binary_search_for_TESTbnd(
                bnd_l=0.0, bnd_u=self.misc["bnd_u"],
                no_binary_search=False, verbose=verbose, max_iter=n_gamma)
            self.misc["bnd_u"] = output["bnd_u"]

        # Otherwise just solve the problem in it's current state
        else:
            self.model.optimize()
            output = self.get_solution()

        model_status = output["model_status"]
        x0_opt = output["x0_opt"]
        distance_squared, propagation_gap, objective_value, certified_radius = \
            output["distance_squared"], output["propagation_gap"], output["objective_value"], output["certified_radius"]

        if verbose >= 2:
            print("  [find_adversarial]: distance", np.sqrt(distance_squared))
            print("  [find_adversarial]: objective value", objective_value)
            print("  [find_adversarial]: certified radius", certified_radius)
            print("  [find_adversarial]: propagation gap", propagation_gap)

        # objective value might be negative if it is almost zero, therefore 'abs'
        if objective_value < 0.0:
            raise Warning(f"  [find_adversarial]: negative objective value {objective_value}")

        res_new = {
            "x_anchor" : self.x_anchor,
            "x_optimal" : x0_opt,
            "y_predicted" : y_predicted,
            "certified_radius" : certified_radius,
            "distance_squared" : distance_squared,
            "objective_value" : objective_value,
            "propagation_gap" : propagation_gap,
            "alpha" : self.alpha,
            "status" : self.model.status,
            "runtime" : time() - t + res[-1]["runtime"]
            }

        res.append(res_new)
        # =============================================================================

        if verbose >= 1:
            print("[find_adversarial]: terminated with")
            print(f"    propagation gap {res[-1]['propagation_gap']} ({eps_compl})")
            print(f"    labels predicted={res[-1]['y_predicted']}, anchor={self.y_anchor}")

        return res




def apply_dbqp(dset, net, prefix, folder=None,
               eps_compl=1e-6, save_each=None, scip_each=0, mask_samples=None, delete=False,
               with_res=True, box_bounds_input=(0.0, 1.0), n_threads=0,
               norm_type="2", mu_type="normal", epsOTHER=None, verbose=1, gurobi=8,
               TESTeach=True, path_alpha=None, n_gamma=10):
    """ Provides a framework for applying QPRel to find the minimal adversarial perturbation.

    Given a classifier and a dataset it computes lower bounds on the distance to the decision boundary from each sample using DBQP.find_adversarial.

    Parameters:
    -----------
    dset : torch.utils.data.Dataset
        Object containing the data (see torch.utils.data and rdl.dataset for details).
    net : rdl.classifier.DenseNet
        NN-classifier to verify (see rdl.classifier for details).
    prefix : str
        Prefix to add to the name of the file containing results.
    eps_compl : float
        See DBQP.find_adversarial (used for computing upper bounds).
    save_each : int or float from (0,1)
        Controls how frequently the results will be written into a file to make them available befor the computation for the whole dataset ends e.g. save_each=0.1 means that results will be saved after processing each 10% of the data (default save_each=None).
    mask_samples : list of bool
        List of the length equal to the number of samples, i's sample will be skipped if mask_samples(i) is False, used to divide the samples between threads in apply_dbqp_parallel (default mask_samples=None).
    delete : bool
        Controls whether the results should be deleted from the memory at the end, if False they will be returned as outputs (default delete=True).
    with_res : bool
        Switches tracking of QPRel stats (outputs from DBQP.find_adversarial()) on or off (default with_res=True).
    box_bounds_input : tuple or None
        Bounds on the admissible adversarials, can be used e.g. to define box constrains for image inputs (default box_bounds_input=None, see DBQP.__init__() for details).
    n_threads : int
        Number of threads allowed for gurobi (default n_threads=0 meaning as many as possible).
    verbose : int
        Verbosity level.

    Returns: (only if delete=False otherwise return None)
    --------
    lb : numpy.ndarray
        Lower bounds on DtDB computed with QPRel
    res_list : list of lists of dicts
        Contains outputs from DBQP.find_adversarial() (filled with None if with_res=False), each res_list[i] corresponding to the i's sample has the following structure
            res_list[i] = [res(1), res(2), ..., res(C)],
        where res(c) is None if 'c' was the anchor label or the output of DBQP.find_adversarial() computed with 'c' as the target label.
    runtime : numpy.ndarray
        Contains wall-clock running time of the whole procedure for each sample (saving the results and initializing the DBQP object are excluded from this measurement).
    mask_samples : same as in the input

    """
    # classes
    classes = range(len(dset.classes))

    n_samples = len(dset)
    if mask_samples is None:
        mask_samples = np.full(len(dset), True)
    n_masked = sum(mask_samples)

    path_bounds =\
        f"dbqp_{dset.name}_" +\
        f"{prefix}_" +\
        f"epscompl{int(-np.log10(eps_compl))}_" +\
        ("" if box_bounds_input is None else f"boxconstr{box_bounds_input[0]}{box_bounds_input[1]}_") +\
        ("" if -1<=scip_each<=1 else f"scip{scip_each}_") +\
        f"res{with_res}_" +\
        f"norm{norm_type}" +\
        ".pkl"
    if folder is None:
        folder = rdl.PATH_RESULTS / "qprel"
    os.makedirs(folder, exist_ok=True)
    path_bounds = folder / path_bounds
    if verbose>0:
        print(f"Results will be saved in\n{path_bounds}\n")

    if type(save_each) is float:
        save_each = int(n_masked*save_each)

    res_list = [None]*n_samples

    # local function writing the current progress into path_bounds
    def save(i, condition=True):
        if condition and save_each != -1:
            if verbose>0:
                print("saved", i, n_samples-1)
            with open(path_bounds, "wb") as f:
                pdump((res_list, mask_samples), f)
        return

    if path_alpha is not None:
        if os.path.exists(path_alpha):
            with open(path_alpha, "rb") as f:
                alpha = pload(f)
            def save_alpha(alpha_):
                pass
        else:
            alpha = None
            def save_alpha(alpha_):
                with open(path_alpha, "wb") as f:
                    pdump(alpha_, f)
    else:
        alpha = None
        def save_alpha(alpha_):
            pass

    # initialize a DBQP object here and later update anchors/labels
    qp = DBQP(net, None, 0, 1,
              alpha=alpha, box_bounds_input=box_bounds_input, n_threads=n_threads,
              norm_type=norm_type, mu_type=mu_type, verbose=verbose, gurobi=gurobi)

    # for resetting alpha later after updating the lables
    alpha_init = qp.alpha
    save_alpha(alpha_init)

    # set counters for the overall number of samples and the number of masked samples
    i_sample = -1
    i_masked = -1

    # =================================================================================
    for i_batch, (X, y) in enumerate(dset.loader):
        X = X.cpu().numpy()
        y = y.cpu().numpy()

        for x_anchor, y_anchor in zip(X, y):
            save(i_sample, condition=((save_each is not None) and\
                                      i_masked > -1 and\
                                      i_masked%save_each == 0))
            i_sample += 1
            i_masked += 1 if (mask_samples is None) else mask_samples[i_sample]


            # SKIPPING CONDITIONS
            ## ignore samples that are not classified by one of the provided classes
            if y_anchor not in classes:
                continue
            ## skip because of non-masked or scip_each
            if (mask_samples is not None and not mask_samples[i_sample]) or\
                    (scip_each > 1 and i_sample%scip_each == 0) or\
                    (scip_each < -1 and i_sample%(-scip_each) != 0):
                mask_samples[i_sample] = False
                continue
            ## skip if misclassified
            if y_anchor != net.labels(x_anchor).item():
                print("[apply_dbqp]: scip misclassified sample.")
                continue


            if with_res:
                res_list[i_sample] = [None]*len(classes)
                res_list[i_sample][y_anchor] = {"target" : [None, None], "runtime" : None}


            if verbose <= 1 or verbose >= 3:
                print(f"\nsample {i_masked}/{n_masked-1} --------------------------")
            t = time()

            # set new anchor
            x_anchor = x_anchor.reshape(-1)
            qp.set_anchor(x_anchor, verbose=verbose)

            lb, ub = inf, inf
            # -------------------------------------------------------------------------
            for y_i, y_target in enumerate(np.random.permutation(classes)):
                if verbose <= 1 or verbose >= 3:
                    print(f"\n  target {y_target} ({y_i+1}/{len(classes)}) --------------------")
                # ignore the case when the anchor label is the target label
                if y_anchor == y_target:
                    continue

                t_ = time()
                if verbose >= 3:
                    print(f"true label {y_anchor}, target label {y_target}")
                    print("computing bound on DtDB...")

                qp.set_labels(y_anchor, y_target, verbose=verbose)
                res = qp.find_adversarial(
                    eps_compl=eps_compl, n_threads=n_threads, verbose=verbose, TESTeach=TESTeach, n_gamma=n_gamma)

                if with_res:
                    res_list[i_sample][y_target] = res
                if verbose >= 3:
                    print(f"{utils.sec2str(time() - t_)}")

                # save the target label that resulted in the smallest bounds
                ## lower bounds
                lb_candidate = res[1]["certified_radius"]

                if lb_candidate < lb:
                    lb = lb_candidate
                    lb_target = y_target
                    res_list[i_sample][y_anchor]["target"][0] = lb_target

            # -------------------------------------------------------------------------
            t = time() - t
            print(utils.sec2str(t))

            res_list[i_sample][y_anchor]["runtime"] = t
            if verbose >= 1:
                print(
                    f"[{i_masked:5}/{n_masked-1:5}] lb : {lb:.4f}, " +
                    f"t : {utils.sec2str(t)}" + "\n")

        del X, y
    # =================================================================================

    save(i_sample)

    if delete:
        del res_list
    return None if delete else res_list


def apply_dbqp_parser(args):
    return apply_dbqp(*args)