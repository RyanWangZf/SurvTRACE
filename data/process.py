import pandas as pd
import pdb
import numpy as np

df = pd.read_csv("seer_raw.csv")

df = df[df["Survival months"] != "Unknown"]
df = df.rename(columns={"Survival months":"duration"})

df = df.drop(["Patient ID"], axis=1)
df = df[df["SEER cause-specific death classification"] != "N/A not seq 0-59"]
df = df[df["Reason no cancer-directed surgery"] != "Not performed, patient died prior to recommended surgery"]

# define features
cat_cols = ["Sex", "Year of diagnosis", "Race recode (W, B, AI, API)", "Histologic Type ICD-O-3",
            "Laterality", "Sequence number", "ER Status Recode Breast Cancer (1990+)",
            "PR Status Recode Breast Cancer (1990+)", "Summary stage 2000 (1998-2017)",
            "RX Summ--Surg Prim Site (1998+)", "Reason no cancer-directed surgery", "First malignant primary indicator",
            "Diagnostic Confirmation", "Median household income inflation adj to 2019"]
num_cols = ["Regional nodes examined (1988+)", "CS tumor size (2004-2015)", "Total number of benign/borderline tumors for patient",
            "Total number of in situ/malignant tumors for patient",]

# special processing
val_counts = df["Histologic Type ICD-O-3"].value_counts()
rank_count = 0
for x in val_counts.items():
    if np.sum(val_counts.values == x[1]) == 1:
        df["Histologic Type ICD-O-3"].replace(x[0], str(rank_count), inplace=True)
        rank_count += 1
    else:
        rep_dict = {k:str(rank_count) for k,v in val_counts[val_counts.values == x[1]].items()}
        df["Histologic Type ICD-O-3"].replace(rep_dict, inplace=True)
        rank_count += 1

# special processing
val_counts = df["Sequence number"].value_counts()
rep_list = [v for v in val_counts[val_counts < 100].index.tolist()]
rep_dict = {k:rep_list[0] for k in rep_list[1:]}
df["Sequence number"].replace(rep_dict, inplace=True)

# special processing
val_counts = df["Diagnostic Confirmation"].value_counts()
rep_list = [v for v in val_counts[val_counts < 160].index.tolist()]
rep_dict = {k:rep_list[0] for k in rep_list[1:]}
df["Diagnostic Confirmation"].replace(rep_dict, inplace=True)

# special processing
df["ER Status Recode Breast Cancer (1990+)"].replace({"Recode not available":"Positive"}, inplace=True)
df["PR Status Recode Breast Cancer (1990+)"].replace({"Recode not available":"Positive"}, inplace=True)
df["Summary stage 2000 (1998-2017)"].replace({"Unknown/unstaged":"Localized"}, inplace=True)
df["Reason no cancer-directed surgery"].replace({"Unknown; death certificate; or autopsy only (2003+)":"Surgery performed"}, inplace=True)
df["Median household income inflation adj to 2019"].replace({"Unknown/missing/no match/Not 1990-2018":"$75,000+"}, inplace=True)

# fill NA, Unknowns
for cat in cat_cols:
    if np.sum(df[cat] == "Unknown") > 0:
        mode_ = df[cat].value_counts().index[0]
        df[cat].replace("Unknown", mode_, inplace=True) 

df_cod_breast = df[df["COD to site recode"] == "Breast"] # 87636
df_cod_heart = df[df["COD to site recode"] == "Diseases of Heart"] # 21584
df_alive = df[df["COD to site recode"] == "Alive"] # 298172

# if heart or alive, event = 0
event_indicator_breast = np.zeros(len(df))
event_indicator_breast[df["COD to site recode"] == "Breast"] = 1
df["event_breast"] = event_indicator_breast

# if breast cancer death or alive, event = 0
event_indicator_heart = np.zeros(len(df))
event_indicator_heart[df["COD to site recode"] == "Diseases of Heart"] = 1
df["event_heart"] = event_indicator_heart

df = pd.concat([
    df[cat_cols], df[num_cols], df["duration"], df["event_heart"], df["event_breast"]
    ], axis=1)

df.to_csv("seer_format.csv", index=False)
print("done first")

from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler

df = pd.read_csv("seer_format.csv")
x_df_cat = {}
for col in cat_cols:
    x_df_cat[col] = LabelEncoder().fit_transform(df[col])
x_df_cat = pd.DataFrame(x_df_cat)

x_num = StandardScaler().fit_transform(df[num_cols])
x_df_num = pd.DataFrame(x_num, columns=num_cols)

df = pd.concat([
    x_df_cat, x_df_num, df["duration"], df["event_heart"], df["event_breast"]
    ], axis=1)
df.to_csv("seer_processed.csv", index=False)


print("Done")