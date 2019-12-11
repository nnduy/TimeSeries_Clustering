import pandas as pd
import numpy as np

#dummy data
date_range = pd.date_range('2017-01-01 00:00', '2017-01-01 00:59', freq='1Min')

print("date_range", date_range)

df = pd.DataFrame(np.random.randint(1, 20, (date_range.shape[0], 1)))
print("df", df)
df.index = date_range  # set index
print("df", df)
df_missing = df.drop(df.between_time('00:12', '00:14').index)
print("df_missing", df_missing)

#check for missing datetimeindex values based on reference index (with all values)
missing_dates = df.index[~df.index.isin(df_missing.index)]

print(missing_dates)
print(type(missing_dates))
print(len(missing_dates))
