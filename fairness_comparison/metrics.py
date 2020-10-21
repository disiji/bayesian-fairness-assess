import math

from metrics_utils import Metric, Average, Diff, Ratio, FilterSensitive
from metrics_utils import calc_pos_protected_percents, calc_prob_class_given_sensitive
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, recall_score


class Accuracy(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'accuracy'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        return accuracy_score(actual, predicted)


class TPR(Metric):
    """
    Returns the true positive rate (aka recall) for the predictions.  Assumes binary
    classification.
    """

    def __init__(self):
        Metric.__init__(self)
        self.name = 'TPR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        return recall_score(actual, predicted, pos_label=positive_pred, average='binary')


class TNR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'TNR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        classes = list(set(actual))
        matrix = confusion_matrix(actual, predicted, labels=classes)
        # matrix[i][j] is the number of observations with actual class i that were predicted as j
        TN = 0.0
        allN = 0.0
        for i in range(0, len(classes)):
            trueval = classes[i]
            if trueval == positive_pred:
                continue
            for j in range(0, len(classes)):
                allN += matrix[i][j]
                predval = classes[j]
                if trueval == predval:
                    TN += matrix[i][j]

        if allN == 0.0:
            return 1.0

        return TN / allN


class FPR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'FPR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        tnr = TNR()
        tnr_val = tnr.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred)
        return 1 - tnr_val


class FNR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'FNR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        tpr = TPR()
        tpr_val = tpr.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred)
        return 1 - tpr_val


class BCR(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'BCR'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        tnr = TNR()
        tnr_val = tnr.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred)
        tpr = TPR()
        tpr_val = tpr.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                           unprotected_vals, positive_pred)
        bcr = (tpr_val + tnr_val) / 2.0
        return bcr


class MCC(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'MCC'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        return matthews_corrcoef(actual, predicted)


class DIBinary(Metric):
    """
    This metric calculates disparate impact in the sense of the 80% rule before the 80%
    threshold is applied.  This is described as DI in: https://arxiv.org/abs/1412.3756
    If there are no positive protected classifications, 0.0 is returned.

    Multiple protected classes are treated as one large group, so that this compares the privileged
    class to all non-privileged classes as a group.
    """

    def __init__(self):
        Metric.__init__(self)
        self.name = 'DIbinary'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        unprotected_pos_percent, protected_pos_percent = \
            calc_pos_protected_percents(predicted, sensitive, unprotected_vals, positive_pred)
        DI = 0.0
        if unprotected_pos_percent > 0:
            DI = protected_pos_percent / unprotected_pos_percent
        if unprotected_pos_percent == 0.0 and protected_pos_percent == 0.0:
            DI = 1.0
        return DI

    def is_better_than(self, val1, val2):
        dist1 = math.fabs(1.0 - val1)
        dist2 = math.fabs(1.0 - val2)
        return dist1 <= dist2


class DIAvgAll(Metric):
    """
    This metric calculates disparate impact in the sense of the 80% rule before the 80%
    threshold is applied.  This is described as DI in: https://arxiv.org/abs/1412.3756
    If there are no positive protected classifications, 0.0 is returned.

    If there are multiple protected classes, the average DI over all groups is returned.
    """

    def __init__(self):
        Metric.__init__(self)
        self.name = 'DIavgall'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        sensitive_values = list(set(sensitive))

        if len(sensitive_values) <= 1:
            print("ERROR: Attempted to calculate DI without enough sensitive values:" + \
                  str(sensitive_values))
            return 1.0

        # this list should only have one item in it
        single_unprotected = [val for val in sensitive_values if val in unprotected_vals][0]
        unprotected_prob = calc_prob_class_given_sensitive(predicted, sensitive, positive_pred,
                                                           single_unprotected)
        sensitive_values.remove(single_unprotected)
        total = 0.0
        for sens in sensitive_values:
            pos_prob = calc_prob_class_given_sensitive(predicted, sensitive, positive_pred, sens)
            DI = 0.0
            if unprotected_prob > 0:
                DI = pos_prob / unprotected_prob
            if unprotected_prob == 0.0 and pos_prob == 0.0:
                DI = 1.0
            total += DI

        if total == 0.0:
            return 1.0

        return total / len(sensitive_values)

    def is_better_than(self, val1, val2):
        dist1 = math.fabs(1.0 - val1)
        dist2 = math.fabs(1.0 - val2)
        return dist1 <= dist2


class CV(Metric):
    def __init__(self):
        Metric.__init__(self)
        self.name = 'CV'

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sensitive = dict_of_sensitive_lists[single_sensitive_name]
        unprotected_pos_percent, protected_pos_percent = \
            calc_pos_protected_percents(predicted, sensitive, unprotected_vals, positive_pred)
        CV = unprotected_pos_percent - protected_pos_percent
        return 1.0 - CV

    def is_better_than(self, val1, val2):
        dist1 = math.fabs(1.0 - val1)
        dist2 = math.fabs(1.0 - val2)
        return dist1 <= dist2


class SensitiveMetric(Metric):
    """
    Takes the given metric and creates a version of it that is conditioned on a sensitive
    attribute.

    For example, for SensitiveMetric(Accuracy), this measure takes the average accuracy per
    sensitive value.  It is unweighted in the sense that each sensitive value's accuracy is treated
    equally in the average.  This measure is designed to catch the scenario when misclassifying all
    Native-Americans but having high accuracy (say, 100%) on everyone else causes an algorithm to
    have 98% accuracy because Native-Americans make up about 2% of the U.S. population.  In this
    scenario, assuming the listed sensitive values were Native-American and not-Native-American,
    this metric would return 1, 0, and 0.5.  Given more than two sensitive values, it will return
    the average over all of the per-value accuracies.  It will also return the ratios with respect
    to the privileged value on this metric, the average of that, the differece with respect to the
    privileged value, and the average of that as well.
    """

    def __init__(self, metric_class):
        Metric.__init__(self)
        self.metric = metric_class
        self.name = self.metric().get_name()  # to be modified as this metric is expanded

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        sfilter = FilterSensitive(self.metric())
        sfilter.set_sensitive_to_filter(self.sensitive_attr, self.sensitive_val)
        return sfilter.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                            unprotected_vals, positive_pred)

    def expand_per_dataset(self, dataset, sensitive_dict, tag):
        objects_list = []
        for sensitive in dataset.get_sensitive_attributes_with_joint():
            objects_list += self.make_metric_objects(sensitive, sensitive_dict, dataset, tag)
        return objects_list

    def make_metric_objects(self, sensitive_name, sensitive_values, dataset, tag):
        privileged_val = self.get_privileged_for_attr(sensitive_name, dataset, tag)

        objs_list = []
        ratios_list = []
        diff_list = []
        for val in sensitive_values[sensitive_name]:
            # adding sensitive variants of the given metric to the objects list
            objs_list.append(self.make_sensitive_obj(sensitive_name, val))

            # adding ratio of sensitive variants of the given metric to the ratios list
            if val != privileged_val:
                ratios_list.append(self.make_ratio_obj(sensitive_name, val, privileged_val))

            # adding diff of sensitive variants of given metric to the diff list
            if val != privileged_val:
                diff_list.append(self.make_diff_obj(sensitive_name, val, privileged_val))
        avg = Average(objs_list, sensitive_name + '-' + self.metric().get_name())
        ratio_avg = Average(ratios_list, sensitive_name + '-' + self.metric().get_name() + 'Ratio')
        diff_avg = Average(diff_list, sensitive_name + '-' + self.metric().get_name() + 'Diff')
        return objs_list + [avg] + ratios_list + [ratio_avg] + diff_list + [diff_avg]

    def make_sensitive_obj(self, sensitive_attr, sensitive_val):
        obj = self.__class__(self.metric)
        obj.set_sensitive_to_filter(sensitive_attr, sensitive_val)
        return obj

    def make_ratio_obj(self, sensitive_attr, sensitive_val, privileged):
        privileged_metric = self.make_sensitive_obj(sensitive_attr, privileged)
        unprivileged_metric = self.make_sensitive_obj(sensitive_attr, sensitive_val)
        return Ratio(unprivileged_metric, privileged_metric)

    def make_diff_obj(self, sensitive_attr, sensitive_val, privileged):
        privileged_metric = self.make_sensitive_obj(sensitive_attr, privileged)
        unprivileged_metric = self.make_sensitive_obj(sensitive_attr, sensitive_val)
        return Diff(privileged_metric, unprivileged_metric)

    def get_privileged_for_attr(self, sensitive_attr, dataset, tag):
        sensitive_attributes = dataset.get_sensitive_attributes_with_joint()
        privileged_vals = dataset.get_privileged_class_names_with_joint(tag)
        for sens, priv in zip(sensitive_attributes, privileged_vals):
            if sens == sensitive_attr:
                return priv
        print("ERROR: couldn't find privileged value for attribute:" + str(sensitive_attr))
        return None

    def set_sensitive_to_filter(self, sensitive_name, sensitive_val):
        """
        Set the attribute and value to filter, i.e., to calculate this metric for.
        """
        self.sensitive_attr = sensitive_name
        self.sensitive_val = sensitive_val
        self.name = str(sensitive_val) + "-" + self.name


##############################
METRICS = [Accuracy(), TPR(), TNR(), BCR(), MCC(),  # accuracy metrics
           DIBinary(), DIAvgAll(), CV(),  # fairness metrics
           # SensitiveMetric(Accuracy), SensitiveMetric(TPR), SensitiveMetric(TNR),
           # SensitiveMetric(FPR), SensitiveMetric(FNR),  # senstive metrics
           ]


def get_metrics(dataset, sensitive_dict, tag):
    """
    Takes a dataset object and a dictionary mapping sensitive attributes to a list of the sensitive
    values seen in the data.  Returns an expanded list of metrics based on the base METRICS.
    """
    metrics = []
    for metric in METRICS:
        metrics += metric.expand_per_dataset(dataset, sensitive_dict, tag)
    return metrics


def add_metric(metric):
    METRICS.append(metric)
