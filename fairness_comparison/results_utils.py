import os
import shutil
import tempfile

from metrics import get_metrics


def get_metrics_list(dataset, sensitive_dict, tag):
    return [metric.get_name() for metric in get_metrics(dataset, sensitive_dict, tag)]


def get_detailed_metrics_header(dataset, sensitive_dict, tag):
    return ','.join(['algorithm', 'params', 'run-id'] + get_metrics_list(dataset, sensitive_dict, tag))


class ResultsFile(object):

    def __init__(self, filename, dataset, sensitive_dict, tag):
        self.filename = filename
        self.dataset = dataset
        self.sensitive_dict = sensitive_dict
        self.tag = tag
        handle, name = self.create_new_file()
        self.fresh_file = handle
        self.tempname = name

    def create_new_file(self):
        fd, name = tempfile.mkstemp()
        os.close(fd)
        f = open(name, "w")
        f.write(get_detailed_metrics_header(
            self.dataset, self.sensitive_dict, self.tag) + '\n')
        return f, name

    def write(self, *args):
        self.fresh_file.write(*args)
        self.fresh_file.flush()
        os.fsync(self.fresh_file.fileno())

    # FIXME close() is terribly inefficient, but :shrug:
    def close(self):
        self.fresh_file.close()

        new_file = open(self.tempname, "r")
        new_columns = new_file.readline().strip().split(',')
        new_rows = new_file.readlines()

        try:
            old_file = open(self.filename, "r")
            old_columns = old_file.readline().strip().split(',')
            old_rows = old_file.readlines()
        except FileNotFoundError:
            old_columns = new_columns[:3]  # copy the key columns
            old_rows = []

        final_columns = set(old_columns).union(set(new_columns))

        # FIXME: here we cross our fingers that parameters don't have "," in them.
        def indexed_rows(rows, column_names):
            result = {}
            for row in rows:
                entries = row.strip().split(',')
                result[tuple(entries[:3])] = dict(
                    (entry_name, entry)
                    for (entry_name, entry) in
                    zip(column_names, entries))
            return result

        old_indexed_rows = indexed_rows(old_rows, old_columns)
        # now we merge the rows onto the old file.
        for (key, value_dict) in indexed_rows(new_rows, new_columns).items():
            for (value_name, value) in value_dict.items():
                old_indexed_rows.setdefault(key, {})[value_name] = value

        fd, final_tempname = tempfile.mkstemp()
        os.close(fd)
        final_file = open(final_tempname, "w")
        final_columns_list = ["algorithm", "params", "run-id"] + \
                             sorted(list(final_columns.difference(set(["algorithm", "params", "run-id"]))))
        final_file.write(",".join(final_columns_list) + "\n")
        for row_dict in old_indexed_rows.values():
            row = ",".join(list(row_dict.get(l, "") for l in final_columns_list))
            final_file.write(row + "\n")
        final_file.close()
        shutil.move(final_tempname, self.filename)


def write_alg_results(file_handle, alg_name, params, run_id, results_list):
    line = alg_name + ','
    params = ";".join("%s=%s" % (k, v) for (k, v) in params.items())
    line += params + (',%s,' % run_id)
    line += ','.join(str(x) for x in results_list) + '\n'
    file_handle.write(line)


def write_predicted_probs(filename, dataset, dataset_obj, predicted_probs_dict):
    print(filename)

    all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
    class_attr = dataset_obj.class_attr
    positive_class_val = dataset_obj.positive_class_val

    dataset = dataset[all_sensitive_attributes + [class_attr]]
    positive_idx = dataset[class_attr] == positive_class_val
    negative_idx = dataset[class_attr] != positive_class_val
    dataset.loc[positive_idx, class_attr] = 1
    dataset.loc[negative_idx, class_attr] = 0

    for algorithm in predicted_probs_dict:
        print("\n\n\nDebug:", algorithm)
        print(predicted_probs_dict[algorithm])
        dataset.loc[:, 'score_' + algorithm] = predicted_probs_dict[algorithm][:, 1]
    dataset.to_csv(filename)


def create_detailed_file(filename, dataset, sensitive_dict, tag):
    return ResultsFile(filename, dataset, sensitive_dict, tag)