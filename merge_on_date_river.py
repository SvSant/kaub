import pandas as pd

filename_sikko = 'data/Waterlevels Sikko.xlsx'
filename_kaub = 'data/water Level Kaub.xlsx'
# filename_soy = 'data/soybeans raw data.xlsx'

kaub = pd.read_excel(filename_kaub, sheet_name='water Level Kaub')

Constance = pd.read_excel(filename_sikko, sheet_name='Sheet1', usecols=['Date-C (GMT)','WL-Constance'])
Maxau = pd.read_excel(filename_sikko, sheet_name='Sheet1', usecols=['Date-W (GMT)','WL Maxau'])
Mainz = pd.read_excel(filename_sikko, sheet_name='Sheet1', usecols=['Date-M (GMT)','MAINZ'])
Line1 = pd.read_excel(filename_sikko, sheet_name='Sheet1', usecols=['Date-L (GMT)','Line'])
Line2 = pd.read_excel(filename_sikko, sheet_name='Sheet1', usecols=['Date-L2 (GMT)','Line2'])

merged = pd.merge(kaub, Constance, how='inner', left_on='datum', right_on='Date-C (GMT)')
merged = pd.merge(merged, Maxau, how='inner', left_on='datum', right_on='Date-W (GMT)')
merged = pd.merge(merged, Mainz, how='inner', left_on='datum', right_on='Date-M (GMT)')
merged = pd.merge(merged, Line1, how='inner', left_on='datum', right_on='Date-L (GMT)')
merged = pd.merge(merged, Line2, how='inner', left_on='datum', right_on='Date-L2 (GMT)')

col = merged.columns.drop(['Date-C (GMT)','Date-W (GMT)','Date-M (GMT)','Date-L (GMT)','Date-L2 (GMT)'])

merged=merged[col]

merged.to_excel('data/merged_all.xlsx')
# with pd.ExcelWriter(filename, mode='a') as writer:
#     merged.to_excel(writer, sheet_name='merged_data')
breakpoint()
