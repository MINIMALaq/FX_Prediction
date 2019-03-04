import pandas as pd
import os

base = './files/all/'
files = os.listdir(base)
fnames = ['dt_end', 'dt_start', 'start', 'label', 'tp', 'before_avg', 'before_density',
          'day_of_week', 'kind', 'red_diff', 'blu_diff', 'diff_to_first', 'result']
for file in files:
    print(file[0:6])
    df = pd.read_csv(base + file, names=fnames, skiprows=1)
    print(df.describe())
