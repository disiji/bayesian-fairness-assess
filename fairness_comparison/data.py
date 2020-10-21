from pathlib import Path

import pandas as pd


def local_results_path():
    current = Path()
    path = current / '.bayesian-fairness'
    if path.exists() and not path.is_dir():
        raise Exception("Cannot run fairness: local storage location %s is not a directory" % path)
    path.mkdir(parents=True, exist_ok=True)
    return path


BASE_DIR = Path('PATH/TO/BASE')
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'preprocessed'
RESULT_DIR = BASE_DIR / 'results'
ANALYSIS_DIR = BASE_DIR / 'analysis'


class Data():
    def __init__(self):
        pass

    def get_dataset_name(self):
        """
        This is the stub name that will be used to generate the processed filenames and is the
        assumed stub for the raw data filename.
        """
        return self.dataset_name

    def get_class_attribute(self):
        """
        Returns the name of the class attribute to be used for classification.
        """
        return self.class_attr

    def get_positive_class_val(self, tag):
        """
        Returns the value used in the dataset to indicate the positive classification choice.
        """
        # FIXME this dependence between tags and metadata is bad; don't know how to fix it right now
        if tag == 'numerical-binsensitive':
            return 1
        else:
            return self.positive_class_val

    def get_sensitive_attributes(self):
        """
        Returns a list of the names of any sensitive / protected attribute(s) that will be used
        for a fairness analysis and should not be used to train the model.
        """
        return self.sensitive_attrs

    def get_sensitive_attributes_with_joint(self):
        """
        Same as get_sensitive_attributes, but also includes the joint sensitive attribute if there
        is more than one sensitive attribute.
        """
        if len(self.get_sensitive_attributes()) > 1:
            return self.get_sensitive_attributes() + ['-'.join(self.get_sensitive_attributes())]
        return self.get_sensitive_attributes()

    def get_privileged_class_names(self, tag):
        """
        Returns a list in the same order as the sensitive attributes list above of the
        privileged class name (exactly as it appears in the data) of the associated sensitive
        attribute.
        """
        # FIXME this dependence between tags and privileged class names is bad; don't know how to
        # fix it right now
        if tag == 'numerical-binsensitive':
            return [1 for x in self.get_sensitive_attributes()]
        else:
            return self.privileged_class_names

    def get_privileged_class_names_with_joint(self, tag):
        """
        Same as get_privileged_class_names, but also includes the joint sensitive attribute if there
        is more than one sensitive attribute.
        """
        priv_class_names = self.get_privileged_class_names(tag)
        if len(priv_class_names) > 1:
            return priv_class_names + ['-'.join(str(v) for v in priv_class_names)]
        return priv_class_names

    def get_categorical_features(self):
        """
        Returns a list of features that should be expanded to one-hot versions for
        numerical-only algorithms.  This should not include the protected features
        or the outcome class variable.
        """
        return self.categorical_features

    def get_features_to_keep(self):
        return self.features_to_keep

    def get_missing_val_indicators(self):
        return self.missing_val_indicators

    def load_raw_dataset(self):
        data_path = self.get_raw_filename()
        data_frame = pd.read_csv(data_path, error_bad_lines=False,
                                 na_values=self.get_missing_val_indicators(),
                                 encoding='ISO-8859-1')
        return data_frame

    def get_raw_filename(self):
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return RAW_DATA_DIR / (self.get_dataset_name() + '.csv')

    def get_filename(self, tag):
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return PROCESSED_DATA_DIR / (self.get_dataset_name() + "_" + tag + '.csv')

    def get_results_filename(self, sensitive_attr, tag):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        return RESULT_DIR / (self.get_dataset_name() + "_" + sensitive_attr + "_" + tag + '.csv')

    def get_predicted_probs_filename(self, sensitive_attr):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        return RESULT_DIR / (self.get_dataset_name() + "_" + sensitive_attr + "_scores.csv")

    def get_param_results_filename(self, sensitive_attr, tag, algname):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        return RESULT_DIR / (algname + '_' + self.get_dataset_name() + "_" + sensitive_attr + \
                             "_" + tag + '.csv')

    def get_analysis_filename(self, sensitive_attr, tag):
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        return ANALYSIS_DIR / (self.get_dataset_name() + "_" + sensitive_attr + "_" + tag + '.csv')

    def data_specific_processing(self, dataframe):
        """
        Takes a pandas dataframe and modifies it to do any data specific processing.  This should
        include any ordered categorical replacement by numbers.  The resulting pandas dataframe is
        returned.
        """
        return dataframe

    def handle_missing_data(self, dataframe):
        """
        This method implements any data specific missing data processing.  Any missing data
        not replaced by values in this step will be removed by the general preprocessing
        script.
        """
        return dataframe

    def get_class_balance_statistics(self, data_frame=None):
        if data_frame is None:
            data_frame = self.load_raw_dataset()
        r = data_frame.groupby(self.get_class_attribute()).size()
        return r

    def get_sensitive_attribute_balance_statistics(self, data_frame=None):
        if data_frame is None:
            data_frame = self.load_raw_dataset()
        return [data_frame.groupby(a).size()
                for a in self.get_sensitive_attributes()]

    ##########################################################################

    def get_results_data_frame(self, sensitive_attr, tag):
        return pd.read_csv(self.get_results_filename(sensitive_attr, tag))

    def get_param_results_data_frame(self, sensitive_attr, tag):
        return pd.read_csv(self.get_param_results_filename(sensitive_attr, tag))


class Ricci(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'ricci'
        # Class attribute will not be created until data_specific_processing is run.
        self.class_attr = 'Class'
        self.positive_class_val = 1
        self.sensitive_attrs = ['Race']
        self.privileged_class_names = ['W']
        self.categorical_features = ['Position']
        self.features_to_keep = ['Position', 'Oral', 'Written', 'Race', 'Combine']
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        dataframe['Class'] = dataframe.apply(passing_grade, axis=1)
        return dataframe

    def handle_missing_data(self, dataframe):
        return dataframe


def passing_grade(row):
    """
    A passing grade in the Ricci data is defined as any grade above a 70 in the combined
    oral and written score.  (See Miao 2010.)
    """
    if row['Combine'] >= 70.0:
        return 1
    else:
        return 0


class Adult(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'adult'
        self.class_attr = 'income-per-year'
        self.positive_class_val = '>50K'
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_class_names = ['White', 'Male']
        self.categorical_features = ['workclass', 'education', 'marital-status', 'occupation',
                                     'relationship', 'native-country']
        self.features_to_keep = ['age', 'workclass', 'education', 'education-num', 'marital-status',
                                 'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                 'capital-loss', 'hours-per-week', 'native-country',
                                 'income-per-year']
        self.missing_val_indicators = ['?']


class German(Data):

    def __init__(self):
        Data.__init__(self)

        self.dataset_name = 'german'
        self.class_attr = 'credit'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sex', 'age']
        self.privileged_class_names = ['male', 'adult']
        self.categorical_features = ['status', 'credit_history', 'purpose', 'savings', 'employment',
                                     'other_debtors', 'property', 'installment_plans',
                                     'housing', 'skill_level', 'telephone', 'foreign_worker']
        self.features_to_keep = ['status', 'month', 'credit_history', 'purpose', 'credit_amount',
                                 'savings', 'employment', 'investment_as_income_percentage',
                                 'personal_status', 'other_debtors', 'residence_since',
                                 'property', 'age', 'installment_plans', 'housing',
                                 'number_of_credits', 'skill_level', 'people_liable_for',
                                 'telephone', 'foreign_worker', 'credit']
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        # adding a derived sex attribute based on personal_status
        sexdict = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                   'A92': 'female', 'A95': 'female'}
        dataframe = dataframe.assign(personal_status= \
                                         dataframe['personal_status'].replace(to_replace=sexdict))
        dataframe = dataframe.rename(columns={'personal_status': 'sex'})

        # adding a derived binary age attribute (youth vs. adult) such that >= 25 is adult
        # this is based on an analysis by Kamiran and Calders
        # http://ieeexplore.ieee.org/document/4909197/
        # showing that this division creates the most discriminatory possibilities.
        old = dataframe['age'] >= 25
        dataframe.loc[old, 'age'] = 'adult'
        young = dataframe['age'] != 'adult'
        dataframe.loc[young, 'age'] = 'youth'
        return dataframe


class PropublicaRecidivism(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'propublica-recidivism'
        self.class_attr = 'two_year_recid'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sex', 'race']
        self.privileged_class_names = ['Male', 'Caucasian']
        self.categorical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']
        # days_b_screening_arrest, score_text, decile_score, and is_recid will be dropped after
        # data specific processing is done
        self.features_to_keep = ["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                                 "juv_other_count", "priors_count", "c_charge_degree",
                                 "c_charge_desc", "decile_score", "score_text", "two_year_recid",
                                 "days_b_screening_arrest", "is_recid"]
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        # Filter as done here:
        # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        dataframe = dataframe[(dataframe.days_b_screening_arrest <= 30) &
                              (dataframe.days_b_screening_arrest >= -30) &
                              (dataframe.is_recid != -1) &
                              (dataframe.c_charge_degree != '0') &
                              (dataframe.score_text != 'N/A')]
        dataframe = dataframe.drop(columns=['days_b_screening_arrest', 'is_recid',
                                            'decile_score', 'score_text'])
        return dataframe


class PropublicaViolentRecidivism(Data):

    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'propublica-violent-recidivism'
        self.class_attr = 'two_year_recid'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sex', 'race']
        self.privileged_class_names = ['Male', 'Caucasian']
        self.categorical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc']
        # days_b_screening_arrest, score_text, decile_score, and is_recid will be dropped after
        # data specific processing is done
        self.features_to_keep = ["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                                 "juv_other_count", "priors_count", "c_charge_degree",
                                 "c_charge_desc", "decile_score", "score_text", "two_year_recid",
                                 "days_b_screening_arrest", "is_recid"]
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        # Filter as done here:
        # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        #
        # The filter for violent recidivism as done above filters out v_score_text instead of
        # score_text, but since we want to include the score_text before the violent recidivism,
        # we think the right thing to do here is to filter score_text.
        dataframe = dataframe[(dataframe.days_b_screening_arrest <= 30) &
                              (dataframe.days_b_screening_arrest >= -30) &
                              (dataframe.is_recid != -1) &
                              (dataframe.c_charge_degree != '0') &
                              (dataframe.score_text != 'N/A')]
        dataframe = dataframe.drop(columns=['days_b_screening_arrest', 'is_recid',
                                            'decile_score', 'score_text'])
        return dataframe


class Credit(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'credit'
        self.class_attr = 'default_payment_next_month'
        self.positive_class_val = 1
        self.sensitive_attrs = ['sex', 'age']
        self.privileged_class_names = [1, 'adult']
        self.categorical_features = ['education', 'marriage']
        self.features_to_keep = ["limit_bal", "sex", "education", "marriage", "age", "pay_0", "pay_2", "pay_3", "pay_4",
                                 "pay_5", "pay_6", "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4", "bill_amt5",
                                 "bill_amt6", "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6",
                                 "default_payment_next_month"]
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        old = dataframe['age'] >= 25
        dataframe.loc[old, 'age'] = 'adult'
        young = dataframe['age'] != 'adult'
        dataframe.loc[young, 'age'] = 'youth'
        dataframe['sex'] = dataframe['sex'].apply(str)
        return dataframe


class Bank(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'bank'
        self.class_attr = 'y'
        self.positive_class_val = 'yes'
        self.sensitive_attrs = ['age']
        self.privileged_class_names = ['adult']
        self.categorical_features = ["job", "marital", "education", "default", "housing", "loan", "contact",
                                     "month", "day_of_week", "poutcome"]
        self.features_to_keep = ["age", "job", "marital", "education", "default", "housing", "loan", "contact",
                                 "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome",
                                 "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "y"]
        self.missing_val_indicators = []

    def data_specific_processing(self, dataframe):
        old = dataframe['age'] >= 25
        dataframe.loc[old, 'age'] = 'adult'
        young = dataframe['age'] != 'adult'
        dataframe.loc[young, 'age'] = 'youth'
        return dataframe


class Census(Data):
    def __init__(self):
        Data.__init__(self)
        self.dataset_name = 'census'

        self.class_attr = 'income-per-year'
        self.positive_class_val = '50000+.'
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_class_names = ['White', 'Male']
        self.categorical_features = ['class-of-worker', 'detailed-industry-recode', 'detailed-occupation-recode',
                                     'education', 'enroll-in-edu-inst-last-wk', 'marital-stat',
                                     'major-industry-code', 'major-occupation-code', 'hispanic-origin',
                                     'member-of-a-labor-union', 'reason-for-unemployment',
                                     'full-or-part-time-employment-stat', 'tax-filer-stat',
                                     'region-of-previous-residence',
                                     'state-of-previous-residence', 'detailed-household-and-family-stat',
                                     'detailed-household-summary-in-household', 'migration-code-change-in-msa',
                                     'migration-prev-res-in-sunbelt',
                                     'country-of-birth-father', 'country-of-birth-mother', 'country-of-birth-self',
                                     'citizenship', 'own-business-or-self-employed',
                                     'fill-inc-questionnaire-for-veterans-admin', 'veterans-benefits']
        self.features_to_keep = ['age', 'class-of-worker', 'detailed-industry-recode', 'detailed-occupation-recode',
                                 'education', 'wage-per-hour', 'enroll-in-edu-inst-last-wk', 'marital-stat',
                                 'major-industry-code', 'major-occupation-code', 'race', 'hispanic-origin', 'sex',
                                 'member-of-a-labor-union', 'reason-for-unemployment',
                                 'full-or-part-time-employment-stat', 'capital-gains', 'capital-losses',
                                 'dividends-from-stocks', 'tax-filer-stat', 'region-of-previous-residence',
                                 'state-of-previous-residence', 'detailed-household-and-family-stat',
                                 'detailed-household-summary-in-household', 'migration-code-change-in-msa',
                                 # 'migration-code-change-in-reg', 'migration-code-move-within-reg', 'live-in-this-house-1-year-ago',
                                 'migration-prev-res-in-sunbelt',
                                 'num-persons-worked-for-employer', 'family-members-under-18',
                                 # 'misc',
                                 'country-of-birth-father', 'country-of-birth-mother', 'country-of-birth-self',
                                 'citizenship', 'own-business-or-self-employed',
                                 'fill-inc-questionnaire-for-veterans-admin', 'veterans-benefits',
                                 'weeks-worked-in-year', 'year', 'income-per-year']
        self.missing_val_indicators = ['?']


###########################################################################
DATASETS = [
    Ricci(),
    Adult(),
    German(),
    PropublicaRecidivism(),
    PropublicaViolentRecidivism(),
    Bank(),
    Credit(),
    # Census(),
]


def get_dataset_names():
    names = []
    for dataset in DATASETS:
        names.append(dataset.get_dataset_name())
    return names


def add_dataset(dataset):
    DATASETS.append(dataset)


def get_dataset_by_name(name):
    for ds in DATASETS:
        if ds.get_dataset_name() == name:
            return ds
    raise Exception("No dataset with name %s could be found." % name)
