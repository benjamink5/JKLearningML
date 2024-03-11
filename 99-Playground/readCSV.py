import pandas as pd
import math

columns = [' splay', ' convergence', ' divergence']
df = pd.read_csv('/Users/okjoonkim/Temp/Validation/tabel_img_right.csv')

print(df.head())
print(df.columns)
metrics_df_fov = df[df[' zone_id'] == -1]

metrics = []
for column in columns:
    metrics_values = metrics_df_fov[column]
    metrics_avg = math.degrees(abs(metrics_values).quantile(0.95))
    metrics.append(metrics_avg)
    
print(metrics[0], metrics[1], metrics[2])