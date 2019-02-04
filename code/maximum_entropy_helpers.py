import numpy as np
from pyswarm import pso
import statsmodels.api as sm


def get_indicator_normalization_constant_constarint(l, points, weights, prior, rules):
    constraint = 0
    indicator_single_rule = []
    for i in range(0, len(rules)):
        rule = rules[i]
        bound_d = rule[0]
        bound_u = rule[1]

        if bound_d == '-inf':
            bound_d = np.min(points) - 1
        if bound_u == 'inf':
            bound_u = np.max(points) + 1

        nearest_l = np.argmin(np.abs(points - bound_d))
        nearest_u = np.argmin(np.abs(points - bound_u))
        indicator = np.append(np.zeros(int(nearest_l)), np.ones(int(nearest_u) - int(nearest_l)))
        indicator = np.append(indicator, np.zeros(len(points) - int(nearest_u)))
        constraint = constraint + (l[i] * indicator)
        indicator_single_rule.append(indicator)
    normalization = np.sum((prior * np.exp(constraint)) * weights)

    return indicator_single_rule, normalization, constraint


def riemann_int(number_of_points, boundary):
    weight = (boundary[1]-boundary[0])/(number_of_points-1)
    points = np.arange(boundary[0], boundary[1], weight)
    points = np.append(points, boundary[1])
    weights = np.ones(number_of_points)*weight

    return points, weights


def max_ent_estimate(l,*args):
    points = args[0]
    weights = args[1]
    prior = args[2]
    rules = args[3]
    indicator_single_rule, normalization, constraint = get_indicator_normalization_constant_constarint(l, points, weights, prior, rules)

    results = []
    for i in range(0, len(rules)):
        rule = rules[i]
        result_lagrange = np.sum((indicator_single_rule[i] * (prior * np.exp(constraint)) * 1/normalization * weights)) - rule[2]
        results.append(result_lagrange)

    return np.sum(np.abs(results))


def get_max_ent_dist(l,points, weights, prior, rules):
    indicator_single_rule, normalization, constraint = get_indicator_normalization_constant_constarint(l, points, weights, prior, rules)
    max_ent_dist = (prior * np.exp(constraint)) * 1 / normalization

    return max_ent_dist


def opt_max_ent(rules,prior_samples):
    kde = sm.nonparametric.KDEUnivariate(np.transpose(np.matrix(prior_samples)))
    kde.fit()
    points, weights = riemann_int(10000, [np.min(prior_samples), np.max(prior_samples)])
    prior = kde.evaluate(points)
    prior = prior/sum(prior*weights)
    lb = np.ones(len(rules))*-20
    ub = np.ones(len(rules))*20

    arguments = (points, weights, prior, rules)
    l, fopt = pso(max_ent_estimate, lb, ub, args=arguments, debug=True, phip=0.5, phig=0.5, omega=0.5,
                           minfunc=1e-12, minstep=0.001, maxiter=100, swarmsize=600)
    max_ent_dist = get_max_ent_dist(l, points, weights, prior, rules)

    return points, weights, prior, max_ent_dist







