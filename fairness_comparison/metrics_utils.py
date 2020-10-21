def calc_pos_protected_percents(predicted, sensitive, unprotected_vals, positive_pred):
    """
    Returns P(C=YES|sensitive=privileged) and P(C=YES|sensitive=not privileged) in that order where
    C is the predicited classification and where all not privileged values are considered
    equivalent.  Assumes that predicted and sensitive have the same lengths.
    """
    unprotected_positive = 0.0
    unprotected_negative = 0.0
    protected_positive = 0.0
    protected_negative = 0.0
    for i in range(0, len(predicted)):
        protected_val = sensitive[i]
        predicted_val = predicted[i]
        if protected_val in unprotected_vals:
            if str(predicted_val) == str(positive_pred):
                unprotected_positive += 1
            else:
                unprotected_negative += 1
        else:
            if str(predicted_val) == str(positive_pred):
                protected_positive += 1
            else:
                protected_negative += 1

    protected_pos_percent = 0.0
    if protected_positive + protected_negative > 0:
        protected_pos_percent = protected_positive / (protected_positive + protected_negative)
    unprotected_pos_percent = 0.0
    if unprotected_positive + unprotected_negative > 0:
        unprotected_pos_percent = unprotected_positive / \
                                  (unprotected_positive + unprotected_negative)

    return unprotected_pos_percent, protected_pos_percent


def calc_confusion_counts(actual, predicted, sensitive, unprotected_vals, positive_pred):
    """
    Returns outs of (C=YES|sensitive=privileged), (C=NO|sensitive=privileged),
    (C=YES|sensitive=not privileged) and (C=NO|sensitive=not privileged) in that order where
    C is the predicited classification and where all not privileged values are considered
    equivalent.  Assumes that predicted and sensitive have the same lengths.
    """

    print("======\n unprotected_val:", unprotected_vals)
    unprotected_TP = 0.0
    unprotected_TN = 0.0
    unprotected_FP = 0.0
    unprotected_FN = 0.0
    protected_TP = 0.0
    protected_TN = 0.0
    protected_FP = 0.0
    protected_FN = 0.0

    for i in range(0, len(predicted)):
        protected_val = sensitive[i]
        predicted_val = predicted[i]
        actual_val = actual[i]
        if protected_val in unprotected_vals:
            if str(predicted_val) == str(positive_pred):
                if str(actual_val) == str(positive_pred):
                    unprotected_TP += 1
                else:
                    unprotected_FP += 1
            else:
                if str(actual_val) == str(positive_pred):
                    unprotected_FN += 1
                else:
                    unprotected_TN += 1
        else:
            if str(predicted_val) == str(positive_pred):
                if str(actual_val) == str(positive_pred):
                    protected_TP += 1
                else:
                    protected_FP += 1
            else:
                if str(actual_val) == str(positive_pred):
                    protected_FN += 1
                else:
                    protected_TN += 1

    return {
        "unprotected_TP": unprotected_TP,
        "unprotected_TN": unprotected_TN,
        "unprotected_FP": unprotected_FP,
        "unprotected_FN": unprotected_FN,
        "protected_TP": protected_TP,
        "protected_TN": protected_TN,
        "protected_FP": protected_FP,
        "protected_FN": protected_FN,
    }


def calc_prob_class_given_sensitive(predicted, sensitive, predicted_goal, sensitive_goal):
    """
    Returns P(predicted = predicted_goal | sensitive = sensitive_goal).  Assumes that predicted
    and sensitive have the same length.  If there are no attributes matching the given
    sensitive_goal, this will error.
    """
    match_count = 0.0
    total = 0.0
    for sens, pred in zip(sensitive, predicted):
        if str(sens) == str(sensitive_goal):
            total += 1
            if str(pred) == str(predicted_goal):
                match_count += 1

    return match_count / total


def calc_fp_fn(actual, predicted, sensitive, unprotected_vals, positive_pred):
    """
    Returns False positive and false negative for protected and unprotected group.
    """
    fp_protected = 0.0
    fp_unprotected = 0.0
    fn_protected = 0.0
    fn_unprotected = 0.0
    for i in range(0, len(predicted)):
        protected_val = sensitive[i]
        predicted_val = predicted[i]
        actual_val = actual[i]
        if protected_val in unprotected_vals:
            if (str(predicted_val) == str(positive_pred)) & (str(actual_val) != str(predicted_val)):
                fp_unprotected += 1
            elif (str(predicted_val) != str(positive_pred)) & (str(actual_val) == str(predicted_val)):
                fn_unprotected += 1
        else:
            if (str(predicted_val) == str(positive_pred)) & (str(actual_val) != str(predicted_val)):
                fp_protected += 1
            elif (str(predicted_val) != str(positive_pred)) & (str(actual_val) == str(predicted_val)):
                fn_protected += 1
    return fp_unprotected, fp_protected, fn_protected, fn_unprotected


class Metric:
    def __init__(self):
        self.name = 'Name not implemented'  ## This should be replaced in implemented metrics.
        self.iter_counter = 0

    def __iter__(self):
        self.iter_counter = 0
        return self

    def __next__(self):
        self.iter_counter += 1
        if self.iter_counter > 1:
            raise StopIteration
        return self

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        """
        actual                          a list of the actual results on the test set
        predicted                       a list of the predicted results
        dict_of_sensitive_lsits         dict mapping sensitive attr names to list of sensitive vals
        single_sensitive_name           sensitive name (dict key) for the sensitive attr being
                                        focused on by this run of the algorithm
        unprotected_vals                a list of the unprotected values for all sensitive attrs
        positive_pred                   the positive value of the prediction task.

        returns                         the calculated result for this metric

        The actual and predicted results and the sensitive attribute lists in the dict should have
        the same length (the length of the test set).

        If there is an error and the metric can not be calculated (e.g., no data is passed in), the
        metric returns None.
        """
        raise NotImplementedError("calc() in Metric is not implemented")

    def get_name(self):
        """
        Returns a name for the metric.  This will be used as the key for a dictionary and will
        also be printed to the final output file.
        """
        return self.name

    def is_better_than(self, val1, val2):
        """
        Compares the two given values that were calculated by this metric and returns true if
        val1 is better than val2, false otherwise.
        """
        return val1 > val2

    def expand_per_dataset(self, dataset, sensitive_dict, tag):
        """
        Optionally allows the expansion of the metric into a returned list of metrics based on the
        dataset, e.g., where there is one metric per sensitive attribute given, and a dictionary
        mapping sensitive attributes to all seen sensitive values from the data.
        """
        return self


class Average(Metric):
    """
    Takes the average (mean) of a given list of metrics.  Assumes that if the total over all
    metrics is 0, the returned result should be 1.
    """

    def __init__(self, metrics_list, name):
        Metric.__init__(self)
        self.name = name
        self.metrics = metrics_list

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):

        total = 0.0
        for metric in self.metrics:
            result = metric.calc(actual, predicted, dict_of_sensitive_lists,
                                 single_sensitive_name, unprotected_vals, positive_pred)
            if result != None:
                total += result

        if total == 0.0:
            return 1.0

        return total / len(self.metrics)

    def is_better_than(self, val1, val2):
        return self.metrics[0].is_better_than(val1, val2)


class Diff(Metric):
    def __init__(self, metric1, metric2):
        Metric.__init__(self)
        self.metric1 = metric1
        self.metric2 = metric2
        self.name = "diff:" + self.metric1.get_name() + 'to' + self.metric2.get_name()

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        m1 = self.metric1.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                               unprotected_vals, positive_pred)
        m2 = self.metric2.calc(actual, predicted, dict_of_sensitive_lists,
                               single_sensitive_name, unprotected_vals, positive_pred)

        if m1 is None or m2 is None:
            return None

        diff = m1 - m2
        return 1.0 - diff

    def is_better_than(self, val1, val2):
        """
        Assumes that 1.0 is the goal value.
        """
        dist1 = math.fabs(1.0 - val1)
        dist2 = math.fabs(1.0 - val2)
        return dist1 <= dist2


class Ratio(Metric):
    def __init__(self, metric_numerator, metric_denominator):
        Metric.__init__(self)
        self.numerator = metric_numerator
        self.denominator = metric_denominator
        self.name = self.numerator.get_name() + '_over_' + self.denominator.get_name()

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):
        num = self.numerator.calc(actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
                                  unprotected_vals, positive_pred)
        den = self.denominator.calc(actual, predicted, dict_of_sensitive_lists,
                                    single_sensitive_name, unprotected_vals, positive_pred)

        if num is None or den is None:
            return None

        if num == 0.0 and den == 0.0:
            return 1.0
        if den == 0.0:
            return 0.0
        return num / den

    def is_better_than(self, val1, val2):
        """
        Assumes that the goal ratio is 1.0.
        """
        dist1 = math.fabs(1.0 - val1)
        dist2 = math.fabs(1.0 - val2)
        return dist1 <= dist2


class FilterSensitive(Metric):
    def __init__(self, metric):
        Metric.__init__(self)
        self.metric = metric
        self.name = metric.get_name()

    def calc(self, actual, predicted, dict_of_sensitive_lists, single_sensitive_name,
             unprotected_vals, positive_pred):

        sensitive = dict_of_sensitive_lists[self.sensitive_for_metric]
        actual_sens = \
            [act for act, sens in zip(actual, sensitive) if sens == self.sensitive_filter]
        predicted_sens = \
            [pred for pred, sens in zip(predicted, sensitive) if sens == self.sensitive_filter]
        sensitive_sens = \
            [sens for sens in sensitive if sens == self.sensitive_filter]

        filtered_dict = {}
        for sens_val in dict_of_sensitive_lists:
            other_sensitive = dict_of_sensitive_lists[sens_val]
            filtered = \
                [s for s, sens in zip(other_sensitive, sensitive) if sens == self.sensitive_filter]
            filtered_dict[sens_val] = filtered

        if len(actual_sens) < 1:
            return None

        return self.metric.calc(actual_sens, predicted_sens, filtered_dict, single_sensitive_name,
                                unprotected_vals, positive_pred)

    def set_sensitive_to_filter(self, sensitive_name, sensitive_val):
        """
        Sets the specific sensitive value to filter based on.  The given metric will be
        calculated only with respect to the actual and predicted values that have this sensitive
        value as part of that item.

        sensitive_name        sensitive attribute name (e.g., 'race')
        sensitive_val         specific sensitive value (e.g., 'white')
        """
        self.name += str(sensitive_val)
        self.sensitive_filter = sensitive_val
        self.sensitive_for_metric = sensitive_name
