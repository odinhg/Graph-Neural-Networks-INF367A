import pandas as pd

#bad_idx = [40357, 57973, 30181, 53798, 11749, 67152, 11751, 64249, 11750, 36815, 70125, 45197]
bad_idx = [70125, 7129]

df = pd.read_pickle("data/time_series_data.pkl")

df2 = pd.DataFrame(columns=df.columns)

for j, i in zip(range(0,2*len(bad_idx), 2), bad_idx):
    df2.loc[j] = df.iloc[i]
    df2.loc[j+1] = df.iloc[i+1]

df2.to_csv("bad_lines.csv")
