from pycox.datasets import metabric, nwtco, support, gbsg, flchain
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import pdb

from .utils import LabelTransform

def load_data(config):
    '''load data, return updated configuration.
    '''
    data = config['data']
    horizons = config['horizons']
    assert data in ["metabric", "nwtco", "support", "gbsg", "flchain", "seer",], "Data Not Found!"
    get_target = lambda df: (df['duration'].values, df['event'].values)

    if data == "metabric":
        # data processing, transform all continuous data to discrete
        df = metabric.read_df()

        # evaluate the performance at the 25th, 50th and 75th event time quantile
        times = np.quantile(df["duration"][df["event"]==1.0], horizons).tolist()

        cols_categorical = ["x4", "x5", "x6", "x7"]
        cols_standardize = ['x0', 'x1', 'x2', 'x3', 'x8']

        df_feat = df.drop(["duration","event"],axis=1)
        df_feat_standardize = df_feat[cols_standardize] 
        df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
        df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

        # must be categorical feature ahead of numerical features!
        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)
        
        vocab_size = 0
        for _,feat in enumerate(cols_categorical):
            df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
            vocab_size = df_feat[feat].max() + 1
                
        # get the largest duraiton time
        max_duration_idx = df["duration"].argmax()
        df_test = df_feat.drop(max_duration_idx).sample(frac=0.3)
        df_train = df_feat.drop(df_test.index)
        df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
        df_train = df_train.drop(df_val.index)

        # assign cuts
        labtrans = LabelTransform(cuts=np.array([df["duration"].min()]+times+[df["duration"].max()]))
        labtrans.fit(*get_target(df.loc[df_train.index]))
        y = labtrans.transform(*get_target(df)) # y = (discrete duration, event indicator)
        df_y_train = pd.DataFrame({"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion": y[2][df_train.index]}, index=df_train.index)
        df_y_val = pd.DataFrame({"duration": y[0][df_val.index], "event": y[1][df_val.index],  "proportion": y[2][df_val.index]}, index=df_val.index)
        # df_y_test = pd.DataFrame({"duration": y[0][df_test.index], "event": y[1][df_test.index],  "proportion": y[2][df_test.index]}, index=df_test.index)
        df_y_test = pd.DataFrame({"duration": df['duration'].loc[df_test.index], "event": df['event'].loc[df_test.index]})
    
    elif data == "support":
        df = support.read_df()
        times = np.quantile(df["duration"][df["event"]==1.0], horizons).tolist()
        cols_categorical = ["x1", "x2", "x3", "x4", "x5", "x6"]
        cols_standardize = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

        df_feat = df.drop(["duration","event"],axis=1)
        df_feat_standardize = df_feat[cols_standardize]        
        df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
        df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)
        
        vocab_size = 0
        for i,feat in enumerate(cols_categorical):
            df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
            vocab_size = df_feat[feat].max() + 1

        # get the largest duraiton time
        max_duration_idx = df["duration"].argmax()
        df_test = df_feat.drop(max_duration_idx).sample(frac=0.3)
        df_train = df_feat.drop(df_test.index)
        df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
        df_train = df_train.drop(df_val.index)

        # assign cuts
        # labtrans = LabTransDiscreteTime(cuts=np.array([0]+times+[df["duration"].max()]))
        labtrans = LabelTransform(cuts=np.array([0]+times+[df["duration"].max()]))

        labtrans.fit(*get_target(df.loc[df_train.index]))
        # y = labtrans.fit_transform(*get_target(df)) # y = (discrete duration, event indicator)
        y = labtrans.transform(*get_target(df)) # y = (discrete duration, event indicator)
        df_y_train = pd.DataFrame({"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion":y[2][df_train.index]}, index=df_train.index)
        df_y_val = pd.DataFrame({"duration": y[0][df_val.index], "event": y[1][df_val.index], "proportion":y[2][df_val.index]}, index=df_val.index)
        # df_y_test = pd.DataFrame({"duration": y[0][df_test.index], "event": y[1][df_test.index], "proportion":y[2][df_test.index]}, index=df_test.index)
        df_y_test = pd.DataFrame({"duration": df['duration'].loc[df_test.index], "event": df['event'].loc[df_test.index]})


    elif data == "seer":
        PATH_DATA = "./data/seer_processed.csv"
        df = pd.read_csv(PATH_DATA)
        times = np.quantile(df["duration"][df["event_breast"]==1.0], horizons).tolist()

        event_list = ["event_breast", "event_heart"]

        cols_categorical = ["Sex", "Year of diagnosis", "Race recode (W, B, AI, API)", "Histologic Type ICD-O-3",
                    "Laterality", "Sequence number", "ER Status Recode Breast Cancer (1990+)",
                    "PR Status Recode Breast Cancer (1990+)", "Summary stage 2000 (1998-2017)",
                    "RX Summ--Surg Prim Site (1998+)", "Reason no cancer-directed surgery", "First malignant primary indicator",
                    "Diagnostic Confirmation", "Median household income inflation adj to 2019"]
        cols_standardize = ["Regional nodes examined (1988+)", "CS tumor size (2004-2015)", "Total number of benign/borderline tumors for patient",
                    "Total number of in situ/malignant tumors for patient",]

        df_feat = df.drop(["duration","event_breast", "event_heart"],axis=1)

        df_feat_standardize = df_feat[cols_standardize]        
        df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
        df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)
        df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)

        vocab_size = 0
        for i,feat in enumerate(cols_categorical):
            df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
            vocab_size = df_feat[feat].max() + 1
        
        # get the largest duraiton time
        max_duration_idx = df["duration"].argmax()
        df_test = df_feat.drop(max_duration_idx).sample(frac=0.3)
        df_train = df_feat.drop(df_test.index)
        df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
        df_train = df_train.drop(df_val.index)

        # assign cuts
        labtrans = LabelTransform(cuts=np.array([0]+times+[df["duration"].max()]))
        get_target = lambda df,event: (df['duration'].values, df[event].values)

        # this datasets have two competing events!
        df_y_train = pd.DataFrame({"duration":df["duration"][df_train.index]})
        df_y_test = pd.DataFrame({"duration":df["duration"][df_test.index]})
        df_y_val = pd.DataFrame({"duration":df["duration"][df_val.index]})

        for i,event in enumerate(event_list):
            labtrans.fit(*get_target(df.loc[df_train.index], event))
            y = labtrans.transform(*get_target(df, event)) # y = (discrete duration, event indicator)

            event_name = "event_{}".format(i)
            df[event_name] = y[1]
            df_y_train[event_name] = df[event_name][df_train.index]
            df_y_val[event_name] = df[event_name][df_val.index]
            df_y_test[event_name] = df[event_name][df_test.index]

        # discretized duration
        df["duration_disc"] = y[0]

        # proportion is the same for all events
        df["proportion"] = y[2]

        df_y_train["proportion"] = df["proportion"][df_train.index]
        df_y_val["proportion"] = df["proportion"][df_val.index]
        df_y_test["proportion"] = df["proportion"][df_test.index]

        df_y_train["duration"] = df["duration_disc"][df_train.index]
        df_y_val["duration"] = df["duration_disc"][df_val.index]
        df_y_test["duration"] = df["duration"][df_test.index]

        # set number of events
        config['num_event'] = 2

    config['labtrans'] = labtrans
    config['num_numerical_feature'] = int(len(cols_standardize))
    config['num_categorical_feature'] = int(len(cols_categorical))
    config['num_feature'] = int(len(df_train.columns))
    config['vocab_size'] = int(vocab_size)
    config['duration_index'] = labtrans.cuts
    config['out_feature'] = int(labtrans.out_features)
    return df, df_train, df_y_train, df_test, df_y_test, df_val, df_y_val