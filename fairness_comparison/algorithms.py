from sklearn.ensemble import RandomForestClassifier as SKLearn_RF
from sklearn.linear_model import LogisticRegression as SKLearn_LR
from sklearn.naive_bayes import GaussianNB as SKLearn_NB
from sklearn.neural_network import MLPClassifier as SKLearn_MLP
from sklearn.svm import SVC as SKLearn_SVM
from sklearn.tree import DecisionTreeClassifier as SKLearn_DT


class Algorithm():
    """
    This is the base class for all implemented algorithms.  New algorithms should extend this
    class, implement run below, and set self.name in the init method.  Other optional methods to
    implement are described below.
    """

    def __init__(self):
        pass

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        """
        Runs the algorithm and returns the predicted classifications on the test set.  The given
        train and test data still contains the sensitive_attrs.  This run of the algorithm
        should focus on the single given sensitive attribute.

        params: a dictionary mapping from algorithm-specific parameter names to the desired values.
        If the implementation of run uses different values, these should be modified in the params
        dictionary as a way of returning the used values to the caller.

        Be sure that the returned predicted classifications are of the same type as the class
        attribute in the given test_df.  If this is not the case, some metric analyses may fail to
        appropriately compare the returned predictions to their desired values.

        TODO: figure out how to indicate that an algorithm that can handle multiple sensitive
        attributes should do so now.
        """
        raise NotImplementedError("run() in Algorithm is not implemented")

    def get_param_info(self):
        """
        Returns a dictionary mapping algorithm parameter names to a list of parameter values to
        be explored.  This function should only be implemented if the algorithm has specific
        parameters that should be tuned, e.g., for trading off between fairness and accuracy.
        """
        return {}

    def get_supported_data_types(self):
        """
        Returns a set of datatypes which this algorithm can process.
        """
        raise NotImplementedError("get_supported_data_types() in Algorithm is not implemented")

    def get_name(self):
        """
        Returns the name for the algorithm.  This must be a unique name, so it is suggested that
        this name is simply <firstauthor>.  If there are mutliple algorithms by the same author(s), a
        suggested modification is <firstauthor-algname>.  This name will appear in the resulting
        CSVs and graphs created when performing benchmarks and analysis.
        """
        return self.name

    def get_default_params(self):
        """
        Returns a dictionary mapping from parameter names to default values that should be used with
        the algorithm.  If not implemented by a specific algorithm, this returns the empty
        dictionary.
        """
        return {}


class Generic(Algorithm):
    def __init__(self):
        Algorithm.__init__(self)
        ## self.classifier should be set in any class that extends this one

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        """
        :param train_df:
        :param test_df:
        :param class_attr:
        :param sensitive_attrs:
        :return: predictions: predicted label on test
        :return: predicted_probs: np.ndarray (num_test, 2). predicted score
        :return: predictions_list: not sure
        """
        # remove sensitive attributes from the training set
        train_df_nosensitive = train_df.drop(columns=sensitive_attrs)
        test_df_nosensitive = test_df.drop(columns=sensitive_attrs)

        # create and train the classifier
        classifier = self.get_classifier()
        y = train_df_nosensitive[class_attr]
        X = train_df_nosensitive.drop(columns=class_attr)
        classifier.fit(X, y)

        # get the predictions on the test set
        X_test = test_df_nosensitive.drop(class_attr, axis=1)
        predictions = classifier.predict(X_test)
        predicted_probs = classifier.predict_proba(X_test)

        return {
            'predictions': predictions,
            'predicted_probs': predicted_probs,
            'predictions_list': [],
        }

    def get_supported_data_types(self):
        # return set(["numerical", "numerical-binsensitive"])
        return set(["numerical"])

    def get_classifier(self):
        """
        Returns the created SKLearn classifier object.
        """
        return self.classifier


class DecisionTree(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_DT()
        self.name = "DecisionTree"


class GaussianNB(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_NB()
        self.name = "GaussianNB"


class LogisticRegression(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_LR()
        self.name = "LR"


class SVM(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_SVM(probability=True)
        self.name = "SVM"


class RandomForest(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_RF(n_estimators=100)
        self.name = 'RandomForest'


class MultiLayerPerceptron(Generic):
    def __init__(self):
        Generic.__init__(self)
        self.classifier = SKLearn_MLP(hidden_layer_sizes=(10,), max_iter=5000)
        self.name = 'MultiLayerPerceptron'


from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
from pandas import DataFrame

REPAIR_LEVEL_DEFAULT = 1.0


class FeldmanAlgorithm(Algorithm):
    def __init__(self, algorithm):
        Algorithm.__init__(self)
        self.model = algorithm
        self.name = 'Feldman-' + self.model.get_name()

    def run(self, train_df, test_df, class_attr, positive_class_val, sensitive_attrs,
            single_sensitive, privileged_vals, params):
        if not 'lambda' in params:
            params = get_default_params()
        repair_level = params['lambda']

        repaired_train_df = self.repair(train_df, single_sensitive, class_attr, repair_level)

        # What should be happening here is that the test_df is transformed using exactly the same
        # transformation as the train_df.  This will only be the case based on the usage below if
        # the distribution of each attribute conditioned on the sensitive attribute is the same
        # in the training set and the test set.
        repaired_test_df = self.repair(test_df, single_sensitive, class_attr, repair_level)

        return self.model.run(repaired_train_df, repaired_test_df, class_attr, positive_class_val,
                              sensitive_attrs, single_sensitive, privileged_vals, params)

    def get_param_info(self):
        """
        Returns lambda values in [0.0, 1.0] at increments of 0.05.
        """
        return {'lambda': [x / 100.0 for x in range(0, 105, 5)]}

    def get_default_params(self):
        return {'lambda': REPAIR_LEVEL_DEFAULT}

    def repair(self, data_df, single_sensitive, class_attr, repair_level):
        types = data_df.dtypes
        data = data_df.values.tolist()

        index_to_repair = data_df.columns.get_loc(single_sensitive)
        headers = data_df.columns.tolist()
        repairer = Repairer(data, index_to_repair, repair_level, False)
        data = repairer.repair(data)

        # The repaired data no longer includes its headers.
        data_df = DataFrame(data, columns=headers)
        data_df = data_df.astype(dtype=types)

        return data_df

    def get_supported_data_types(self):
        """
        The Feldman algorithm can preprocess both numerical and categorical data, the limiting
        factor is the capacity of the model that data is then passed to.
        """
        return self.model.get_supported_data_types()

    def binary_sensitive_attrs_only(self):
        return False


################################################
ALGORITHMS = [
    # SVM(), SVM takes too long, in order to predict probability
    GaussianNB(),
    LogisticRegression(), DecisionTree(), RandomForest(), MultiLayerPerceptron(),  # baseline
    # FeldmanAlgorithm(SVM()), 
    FeldmanAlgorithm(GaussianNB()),  # Feldman
    FeldmanAlgorithm(LogisticRegression()), FeldmanAlgorithm(DecisionTree()),
    FeldmanAlgorithm(RandomForest()), FeldmanAlgorithm(MultiLayerPerceptron()),
]


def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result
