import pandas as pd


def get_data(name):
    filename = 'data/raw_data/'+name+'.csv'
    data = pd.read_csv(filename)
    data = data.rename(columns = {'avg':name})[['shape_id', 'weather_date', name]]
    data.set_index('weather_date', inplace=True)

    return data

def merge_data(left, right):
    left_right = left.merge(right, how='inner', left_on=['weather_date', 'shape_id'], right_on=['weather_date', 'shape_id'])
    return left_right


minT = get_data('minT')
maxT = get_data('maxT')
meanT = get_data('meanT')
precip = get_data('precip')
snow = get_data('snow')
soil_5 = get_data('soil_5')
soil_15 = get_data('soil_15')

merged = merge_data(minT, maxT)
merged = merge_data(merged, meanT)
merged = merge_data(merged, precip)
merged = merge_data(merged, snow)
merged = merge_data(merged, soil_5)
merged = merge_data(merged, soil_15)

# merged.to_excel('data/merged_weather.xlsx')

shapes = merged['shape_id'].unique()

for i in range(len(shapes)):
    shape_data = merged.loc[merged['shape_id'] == shapes[i]]
    col = shape_data.columns[1:]
    
    shape_data = shape_data[col]
    idx = shapes[i] - 340000
    
    shape_data = shape_data.add_prefix(str(idx)+'_')

    if i == 0:
        all_data = shape_data
    else:
        all_data = all_data.merge(shape_data, how='inner', left_on='weather_date', right_on='weather_date')

all_data.to_excel('data/all_data.xlsx')

breakpoint()
merged = merged.reset_index()
merged = merged.set_index(['weather_date','shape_id']).unstack().swaplevel(0,1,1).sort_index(axis=1)
breakpoint()
merged.to_excel('data/merged_weather_shape_seperated.xlsx')
breakpoint()
