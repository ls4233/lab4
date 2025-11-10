import pandas as pd
import numpy as np
import wfdb
import ast


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Initialization
local_path = "/Users/leelasrinivasan/Desktop/SignalLab/lab4_files/data/"
sampling_rate=100


# Load and converlst annotation data
Y = pd.read_csv(local_path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


# Load raw signal data
X = load_raw_data(Y, sampling_rate, local_path)


# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(local_path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)


# Run stats on data
Y['class'] = Y['diagnostic_superclass'].str[0] # Flatten lists
class_counts = Y['class'].value_counts()


# Filter out unrealistic age counts
Y = Y[(Y['age'] > 0) & (Y['age'] < 110)]
avg_age = Y.groupby('class')['age'].mean()
std_age = Y.groupby('class')['age'].std()


# Breakdown by heart_axis
counts = Y.groupby(['class', 'heart_axis']).size().unstack(fill_value=0)


# Split data into train and test
test_fold = 10
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
