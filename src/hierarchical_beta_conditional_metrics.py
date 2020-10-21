import argparse
import os
import pickle

import numpy as np
from scipy.stats import beta
from utils import *

num_samples = 800

DIR = 'PATH/TO/DATA/'
OUTPUT_DIR = 'PATH/TO/OUTPUT/'
FIG_DIR = 'PATH/TO/FIGURES/'


def BetaCalibration(s, a, b, c):
    num_scores = s.shape[0]

    # p is the model probability of label "1" given this score
    P = np.zeros((num_scores, num_samples))

    for i in range(num_samples):
        logoddsp = c[i] + np.exp(a[i]) * np.log(s) - np.exp(b[i]) * np.log(1 - s)
        logoddsp[logoddsp > 15] = 15
        P[:, i] = np.exp(logoddsp) / (np.exp(logoddsp) + 1)

    return P


def predict_groupwise(y_hat, p, ct, num_groups):
    """

    :param y_hat: (N, ). Prediction from M, binary
    :param p: (N, num_samples) calibrated model probability of label "1"
    :param ct: (N, ) group membership fo each instance
    :param num_groups: int
    :return:
    """

    tpr = np.zeros((num_groups, num_samples))
    fpr = np.zeros((num_groups, num_samples))
    tnr = np.zeros((num_groups, num_samples))
    fnr = np.zeros((num_groups, num_samples))

    pr = np.zeros((num_groups, num_samples))
    nr = np.zeros((num_groups, num_samples))

    for k in range(num_groups):
        p_k = p[ct == k]
        y_hat_k = y_hat[ct == k]

        # p_base[i] = p(y_hat=i)
        p_base = np.zeros((2, num_samples))
        p_base[0] = np.mean((y_hat_k == 0), axis=0)
        p_base[1] = np.mean((y_hat_k == 1), axis=0)

        # p_condition[i,j] = p(y=i|y_hat=j)
        p_condition = np.zeros((2, 2, num_samples))
        p_condition[1, 0] = np.mean(p_k[y_hat_k == 0], axis=0)
        p_condition[1, 1] = np.mean(p_k[y_hat_k == 1], axis=0)
        p_condition[0, 0] = 1 - p_condition[1, 0]
        p_condition[0, 1] = 1 - p_condition[1, 1]
        # important!
        p_condition[np.isnan(p_condition)] = 0

        # p_marginal[i] = p(y = 1)
        p_marginal = np.zeros((2, num_samples))
        p_marginal[0] = p_base[0] * p_condition[0, 0] + p_base[1] * p_condition[0, 1]
        p_marginal[1] = p_base[0] * p_condition[1, 0] + p_base[1] * p_condition[1, 1]

        # compute measurement
        tpr[k] = p_base[1] * p_condition[1, 1] / p_marginal[1]
        fpr[k] = 1 - tpr[k]
        tnr[k] = p_base[0] * p_condition[0, 0] / p_marginal[0]
        fnr[k] = 1 - tnr[k]

        pr[k] = p_marginal[1]
        nr[k] = p_marginal[0]

    return {
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'pr': pr,
        'nr': nr
    }


def compute_groupwise(y_hat, y, ct, num_groups):
    """

    :param y_hat: (N, ). Prediction from M, binary
    :param y: (N, ) true label
    :param ct: (N, ) group membership fo each instance
    :param num_groups: int
    :return:
    """
    tpr = np.zeros((num_groups,))
    fpr = np.zeros((num_groups,))
    tnr = np.zeros((num_groups,))
    fnr = np.zeros((num_groups,))

    pr = np.zeros((num_groups,))
    nr = np.zeros((num_groups,))

    for k in range(num_groups):
        y_hat_k = y_hat[ct == k]
        y_k = y[ct == k]
        tpr[k] = y_hat_k[y_k == 1].mean()
        fpr[k] = 1 - tpr[k]
        tnr[k] = 1 - y_hat_k[y_k == 0].mean()
        fnr[k] = 1 - tnr[k]

        pr[k] = y.mean()
        nr[k] = 1 - y.mean()

    return {
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'pr': pr,
        'nr': nr
    }


def beta_binomial_groupwise(y_hat, y, ct, num_groups, alpha_0, beta_0):
    """
    :param y_hat: (N, ). Prediction from M, binary
    :param y: (N, ) true label
    :param ct: (N, ) group membership fo each instance
    :param num_groups: int
    :return: dict, each metric of shape (num_groups, num_samples)
    """
    tpr = np.zeros((num_groups, num_samples))
    fpr = np.zeros((num_groups, num_samples))
    tnr = np.zeros((num_groups, num_samples))
    fnr = np.zeros((num_groups, num_samples))

    pr = np.zeros((num_groups, num_samples))
    nr = np.zeros((num_groups, num_samples))

    # beta_binomial_groupwise for each metric, prior beta(0.5,0.5) for each group
    params_tpr = np.ones((num_groups, 2))
    params_tpr[:, 0] *= alpha_0
    params_tpr[:, 1] *= beta_0

    params_fpr = np.ones((num_groups, 2))
    params_fpr[:, 0] *= alpha_0
    params_fpr[:, 1] *= beta_0

    params_tnr = np.ones((num_groups, 2))
    params_tnr[:, 0] *= alpha_0
    params_tnr[:, 1] *= beta_0

    params_fnr = np.ones((num_groups, 2))
    params_fnr[:, 0] *= alpha_0
    params_fnr[:, 1] *= beta_0

    params_pr = np.ones((num_groups, 2))
    params_pr[:, 0] *= alpha_0
    params_pr[:, 1] *= beta_0

    params_nr = np.ones((num_groups, 2))
    params_nr[:, 0] *= alpha_0
    params_nr[:, 1] *= beta_0

    for k in range(num_groups):
        y_hat_k = y_hat[ct == k]
        y_k = y[ct == k]

        params_tpr[k, 0] += y_hat_k[y_k == 1].sum()
        params_tpr[k, 1] += (1 - y_hat_k[y_k == 1]).sum()
        params_fpr[k, 0] += (1 - y_hat_k[y_k == 1]).sum()
        params_fpr[k, 1] += y_hat_k[y_k == 1].sum()

        params_tnr[k, 0] += (1 - y_hat_k[y_k == 0]).sum()
        params_tnr[k, 1] += y_hat_k[y_k == 0].sum()
        params_fnr[k, 0] += y_hat_k[y_k == 0].sum()
        params_fnr[k, 1] += (1 - y_hat_k[y_k == 0]).sum()

        params_pr[k, 0] += (1 - y_k).sum()
        params_pr[k, 1] += y_k.sum()
        params_nr[k, 0] += y_k.sum()
        params_nr[k, 1] += (1 - y_k).sum()

        tpr[k] = beta.rvs(params_tpr[k, 0], params_tpr[k, 1], size=(num_samples,))
        fpr[k] = beta.rvs(params_fpr[k, 0], params_fpr[k, 1], size=(num_samples,))
        tnr[k] = beta.rvs(params_tnr[k, 0], params_tnr[k, 1], size=(num_samples,))
        fnr[k] = beta.rvs(params_fnr[k, 0], params_fnr[k, 1], size=(num_samples,))
        pr[k] = beta.rvs(params_pr[k, 0], params_pr[k, 1], size=(num_samples,))
        nr[k] = beta.rvs(params_nr[k, 0], params_nr[k, 1], size=(num_samples,))

    return {
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'pr': pr,
        'nr': nr
    }


def main():
    experiment_name = '%s_%s_budget%d' % (args.dataset, args.attribute, args.budget)

    path = OUTPUT_DIR + 'hierarchical_beta_calibration' + '/' + experiment_name + '/'

    metrics = dict()  # [method, algorithm, run_id, metric, group_id, num_samples]

    metrics['hierarchical_beta_calibration'] = {}
    metrics['frequentist'] = {}
    metrics['beta_binomial_1'] = {}
    metrics['beta_binomial_05'] = {}
    metrics['beta_binomial_01'] = {}
    metrics['beta_binomial_001'] = {}
    metrics['ground_truth'] = {}

    for idx, algorithm in enumerate(algorithms):

        # load calibration parameters
        sampled_a = np.zeros((args.num_runs, args.num_groups, num_samples))
        sampled_b = np.zeros((args.num_runs, args.num_groups, num_samples))
        sampled_c = np.zeros((args.num_runs, args.num_groups, num_samples))
        for run_id in range(args.num_runs):
            tmp = pickle.load(open(path + 'samples_run%d.pkl' % run_id, 'rb'))[algorithm]
            sampled_a[run_id] = tmp['a'].reshape((args.num_groups, num_samples))
            sampled_b[run_id] = tmp['b'].reshape((args.num_groups, num_samples))
            sampled_c[run_id] = tmp['c'].reshape((args.num_groups, num_samples))

        # calibrate and compute metrics
        metrics['hierarchical_beta_calibration'][algorithm] = {}
        metrics['frequentist'][algorithm] = {}
        metrics['beta_binomial_1'][algorithm] = {}
        metrics['beta_binomial_05'][algorithm] = {}
        metrics['beta_binomial_01'][algorithm] = {}
        metrics['beta_binomial_001'][algorithm] = {}

        dataset = Dataset.load_from_file(DIR + "%s_%s_scores_remapped.csv" % \
                                         (args.dataset, args.attribute), args.dataset)
        s = dataset.df['score_%s' % algorithm]
        y = dataset.df[dataset.class_attr]
        y_hat = (s > 0.5) * 1.0
        metrics['ground_truth'][algorithm] = compute_groupwise(y_hat, y, dataset.df[args.attribute], args.num_groups)

        for run_id in tqdm(range(args.num_runs)):

            dataset = Dataset.load_from_file(DIR + "%s_%s_scores_remapped.csv" % \
                                             (args.dataset, args.attribute), args.dataset)
            dataset.shuffle(random_state=run_id, attribute=args.attribute)

            s = dataset.df['score_%s' % algorithm]
            y = dataset.df[dataset.class_attr]
            y_hat = (s > 0.5) * 1.0

            # Groupwise beta calibraiton: compute z_i
            # p is the model probability of label "1" given this score, i.e. calibrated s
            p = np.zeros((dataset.__len__(), num_samples))
            for k in range(args.num_groups):
                mask = dataset.df[args.attribute] == k
                p[mask, :] = BetaCalibration(s[mask],
                                             sampled_a[run_id, k],
                                             sampled_c[run_id, k],
                                             sampled_c[run_id, k])
            p[:args.budget] = y[:args.budget:, np.newaxis]
            metrics['hierarchical_beta_calibration'][algorithm][run_id] = predict_groupwise(y_hat,
                                                                                            p,
                                                                                            dataset.df[args.attribute],
                                                                                            args.num_groups)

            # Frequentist_estimation
            metrics['frequentist'][algorithm][run_id] = compute_groupwise(y_hat[:args.budget],
                                                                          y[:args.budget],
                                                                          dataset.df[args.attribute][:args.budget],
                                                                          args.num_groups)

            # Beta Bernoulli samples
            metrics['beta_binomial_1'][algorithm][run_id] = beta_binomial_groupwise(y_hat[:args.budget],
                                                                                    y[:args.budget],
                                                                                    dataset.df[args.attribute][
                                                                                    :args.budget],
                                                                                    args.num_groups,
                                                                                    1, 1)
            # Beta Bernoulli samples
            metrics['beta_binomial_05'][algorithm][run_id] = beta_binomial_groupwise(y_hat[:args.budget],
                                                                                     y[:args.budget],
                                                                                     dataset.df[args.attribute][
                                                                                     :args.budget],
                                                                                     args.num_groups,
                                                                                     0.5, 0.5)
            # Beta Bernoulli samples
            metrics['beta_binomial_01'][algorithm][run_id] = beta_binomial_groupwise(y_hat[:args.budget],
                                                                                     y[:args.budget],
                                                                                     dataset.df[args.attribute][
                                                                                     :args.budget],
                                                                                     args.num_groups,
                                                                                     0.1, 0.1)
            # Beta Bernoulli samples
            metrics['beta_binomial_001'][algorithm][run_id] = beta_binomial_groupwise(y_hat[:args.budget],
                                                                                      y[:args.budget],
                                                                                      dataset.df[args.attribute][
                                                                                      :args.budget],
                                                                                      args.num_groups,
                                                                                      0.01, 0.01)

    # write files
    path = OUTPUT_DIR + 'conditional_metrics/' + experiment_name + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    for item in metrics:
        pickle.dump(metrics[item], open(path + '%s.pkl' % item, 'wb'), -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str, default='adult',
                        help='dataset name: race, bank, propublica-violent-recidivism')
    parser.add_argument('attribute', type=str, default='sex',
                        help='attribute  used to group instances')
    parser.add_argument('num_groups', type=int, default=2, help='number of distinct values of sensitive attribute')
    parser.add_argument('num_runs', type=int, default=50, help='number of runs')
    parser.add_argument('budget', type=int, default=10,
                        help='total number of samples to label, across different groups')
    args, _ = parser.parse_known_args()

    main()
