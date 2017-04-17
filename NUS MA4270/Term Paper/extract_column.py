import pandas as pd

df = pd.read_csv('Speed Dating Data.csv')
df = df.loc[:, ['iid', 'gender', 'match', 'age_o', 'race_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb', 'pf_o_sha', 'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'age', 'race', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr', 'sinc', 'intel', 'fun', 'amb', 'shar']]

df.to_csv('Selected_Column_Data.csv', index=False, encoding='utf-8')