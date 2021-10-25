import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pyjags
import xarray as xr
from utils import *

# Load additional JAGS module
pyjags.load_module('glm')
plt.style.use('ggplot')

DIAGNOSIS = False

DIR = '../results/'
OUTPUT_DIR = '../results/'


def main():
    experiment_name = '%s_%s_budget%d' % (args.dataset, args.attribute, args.budget)
    path = OUTPUT_DIR + experiment_name

    if not os.path.exists(path):
        os.makedirs(path)

    dataset = Dataset.load_from_file(DIR + "%s_%s_scores_remapped.csv" % (args.dataset, args.attribute), args.dataset)
    print("\n\n\n================%s================" % dataset.dataset_name)

    dataset.shuffle(random_state=args.run_id, attribute=args.attribute)

    n = dataset.df.shape[0]
    vars_list = ['theta', 'mu', 'c']

    # train the model
    samples = dict()
    for idx, algorithm in enumerate(algorithms):
        print("=================%s=================" % algorithm)
        model = pyjags.Model(code_hierarchical_beta_binomial,
                             data=dict(nl=args.budget,
                                       nc=args.num_groups,
                                       ct=dataset.df[args.attribute][: n] + 1,
                                       s=np.clip(dataset.df['score_' + algorithm][: n], 0.01, 0.99),
                                       y=dataset.df[dataset.class_attr][: n]),
                             chains=4, adapt=1000)
        model.sample(500, vars=vars_list)
        samples[algorithm] = model.sample(200, vars=vars_list)

    pickle.dump(samples, open(path + '/samples_run%d.pkl' % args.run_id, 'wb'), -1)

    for idx, algorithm in enumerate(algorithms):
        print("=================%s=================" % algorithm)
        for group_id in range(args.num_groups):
            print("======Group:%d======" % group_id)
            trace = xr.Dataset({k: (("Iteration", "Chain"), v[group_id]) for \
                                k, v in samples[algorithm].items() if k not in {'mu', 'c'}})

            print(trace.to_dataframe().mean())
            print(trace.to_dataframe().quantile([0.05, 0.95]))
            # True accuracy

            mask = (dataset.df[args.attribute] == group_id)

            # all instaces
            y = dataset.df[dataset.class_attr][:n][mask]
            s = dataset.df['score_' + algorithm][:n][mask]
            predlabel = (s > 0.5) * 1.0
            s = np.maximum(s, 1 - s)

            # labeled instaces
            yl = dataset.df[dataset.class_attr][:args.budget][mask]
            sl = dataset.df['score_' + algorithm][:args.budget][mask]
            predlabell = (sl > 0.5) * 1.0

            sampled_theta = samples[algorithm]['theta'][group_id].flatten()
            mean_theta = np.mean(sampled_theta)
            lb_theta = np.quantile(sampled_theta, 0.025)
            ub_theta = np.quantile(sampled_theta, 0.975)
            print("========  True accuracy:                   %.2f" % (y == predlabel).mean())
            print("========  Predicted accuracy:   %.2f, (%.2f, %.2f)" %
                  (mean_theta, lb_theta, ub_theta))
            print("========  Empirical  accuracy:             %.2f" % (yl == predlabell).mean())
            print("========  Uncalibrated predicted  accuracy:%.2f\n" % (s.mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default='adult',
                        help='dataset name: race, bank, propublica-violent-recidivism')
    parser.add_argument('attribute', type=str, default='sex',
                        help='attribute  used to group instances')
    parser.add_argument('num_groups', type=int, default=2, help='number of distinct values of sensitive attribute')
    parser.add_argument('budget', type=int, default=10,
                        help='total number of samples to label, across different groups')
    parser.add_argument('run_id', type=int, default=0, help='random state for dataset.shuffle()')
    args, _ = parser.parse_known_args()

    main()
