from copy import deepcopy

import fire
from algorithms import ALGORITHMS
from bayesian_metrics import get_bayesian_metrics
from metrics import get_metrics
from preprocess import ProcessedData
from results_utils import create_detailed_file, write_alg_results, write_predicted_probs

from data import DATASETS, get_dataset_names

NUM_TRIALS_DEFAULT = 1


def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result


def run(num_trials=NUM_TRIALS_DEFAULT, dataset=get_dataset_names(),
        algorithm=get_algorithm_names()):
    """
    - Step 1: load preprocessed data and split it into train and test by 2/3 and 1/3
    - Step 2: train and evaluate algorithm by calling `run_eval_alg`
    - Step 3: write results (metrics eval and predicted probs) to file

    :param num_trials:
    :param dataset:
    :param algorithm:
    :return:
    """

    algorithms_to_run = algorithm
    print(algorithms_to_run)

    print("Datasets: %s" % dataset)
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("Evaluating dataset:" + dataset_obj.get_dataset_name())

        processed_dataset = ProcessedData(dataset_obj)
        # train_test_splits: a dict maps key to a list of (train, test) tuple of length num_trials
        train_test_splits = processed_dataset.create_train_test_splits(num_trials)

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes()  # dataset_obj.get_sensitive_attributes_with_joint()

        print("All sensitive attributes:" + ",".join(all_sensitive_attributes))

        for sensitive in all_sensitive_attributes:
            print("Sensitive attribute:" + sensitive)

            detailed_files = dict((k, create_detailed_file(
                dataset_obj.get_results_filename(sensitive, k),
                dataset_obj,
                processed_dataset.get_sensitive_values(k), k))
                                  for k in train_test_splits.keys())

            for i in range(0, num_trials):
                predicted_probs_dict = {}

                for algorithm in ALGORITHMS:

                    print("\n\n", algorithm.name)
                    if not algorithm.get_name() in algorithms_to_run:
                        print("!!!!!!!!!!")
                        continue

                    for supported_tag in algorithm.get_supported_data_types():
                        train, test = train_test_splits[supported_tag][i]  # train and test are pandas.DataFrame
                        try:
                            eval_output = run_eval_alg(algorithm, train, test, dataset_obj, processed_dataset,
                                                       all_sensitive_attributes, sensitive, supported_tag)
                        except Exception as e:
                            print("Failed: %s" % e)
                        else:
                            write_alg_results(detailed_files[supported_tag],
                                              algorithm.get_name(), eval_output['params'], i,
                                              eval_output['metrics_eval'])
                            predicted_probs_dict[algorithm.name] = eval_output['test_prediction_probs']

                if i == 0:
                    # write the predicted probs on testset to file
                    filename = dataset_obj.get_predicted_probs_filename(sensitive)
                    write_predicted_probs(filename, test, dataset_obj, predicted_probs_dict)

            print("Results written to:")
            for supported_tag in algorithm.get_supported_data_types():
                print("    %s" % dataset_obj.get_results_filename(sensitive, supported_tag))
            print("    %s" % dataset_obj.get_predicted_probs_filename(sensitive))

            for detailed_file in detailed_files.values():
                detailed_file.close()


def run_eval_alg(algorithm, train, test, dataset, processed_data, all_sensitive_attributes,
                 single_sensitive, tag):
    """
    Train and evaluate the algorithm; and gets the resulting metric evaluations.

    :return: params

    """
    privileged_vals = dataset.get_privileged_class_names_with_joint(tag)
    positive_val = dataset.get_positive_class_val(tag)

    ##############TRAIN CLASSIFICATION MODEL ON TRAIN AND PREDICT ON TEST
    # get the actual classifications and sensitive attributes
    actual = test[dataset.get_class_attribute()].values.tolist()

    alg_output = run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
                         privileged_vals, positive_val)

    ##############EVAL AGGREGATED METRICS ON TEST SET
    # make dictionary mapping sensitive names to sensitive attr test data lists
    dict_sensitive_lists = {}
    for sens in all_sensitive_attributes:
        dict_sensitive_lists[sens] = test[sens].values.tolist()

    sensitive_dict = processed_data.get_sensitive_values(tag)
    metrics_eval = []
    for metric in get_metrics(dataset, sensitive_dict, tag):
        result = metric.calc(actual, alg_output['predictions'], dict_sensitive_lists, single_sensitive,
                             privileged_vals, positive_val)
        metrics_eval.append(result)
    for metric in deepcopy(get_bayesian_metrics(dataset, sensitive_dict, tag)):
        result = metric.calc(actual, alg_output['predictions'], dict_sensitive_lists, single_sensitive,
                             privileged_vals, positive_val)
        metrics_eval.append(result)

    results_lol = []

    return {
        'params': alg_output['params'],
        'metrics_eval': metrics_eval,
        'results_lol': results_lol,
        'test_prediction_probs': alg_output['predicted_probs']
    }


def run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
            privileged_vals, positive_val):
    """

    :param algorithm:
    :param train:
    :param test:
    :param dataset:
    :param all_sensitive_attributes:
    :param single_sensitive:
    :param privileged_vals:
    :param positive_val:

    :return: predictions: np.ndarray (num_test, ), predicted class of the model, INSTEAD OF CONFIDENCE
    :return: predicted_probs: np.ndarray (num_test, num_classes), confidence of the model
    :return: params: {} unless this is default params specified for the model
    :return: predictions_list: [] unless the algorithm is fairness aware
    """
    class_attr = dataset.get_class_attribute()
    all_sensitive_attributes = dataset.get_sensitive_attributes_with_joint()
    params = algorithm.get_default_params()

    # Note: the training and test set here still include the sensitive attributes because
    # some fairness aware algorithms may need those in the dataset.  They should be removed
    # before any model training is done.

    ######################
    # TODO: in the training process, senstive attributes are removed. what is the proper way to handle this?
    # predictions_list is [] if the model is not fairness aware
    alg_output = algorithm.run(train, test, class_attr, positive_val, all_sensitive_attributes,
                               single_sensitive, privileged_vals, params)
    alg_output['params'] = params

    return alg_output


def main():
    fire.Fire(run)


if __name__ == '__main__':
    main()
