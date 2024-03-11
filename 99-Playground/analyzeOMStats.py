import numpy as np
import pandas as pd

df = pd.read_csv('./Data/statsCSV.csv')
# print(df.head())
# print(df.describe())
# print(df.info())
# print(df.columns)
'''
['blemishID', 'lensZoneID', 'lensZonePass',
       'lensDefectSpecLimit_um_lower', 'lensDefectSpecLimit_um_upper',
       'displayZoneID', 'dispZonePass', 'dispDefectSpecLimit_um_lower',
       'dispDefectSpecLimit_um_upper', 'cleanable', 'dispZoneAngle_deg',
       'dispZoneRadius_pix', 'camCentroid_pix_X', 'camCentroid_pix_Y',
       'dispCoords_pix_X', 'dispCoords_pix_Y', 'dispCoords_mm_X',
       'dispCoords_mm_Y', 'contrastInCameraImage', 'surfaceIndex',
       'surfaceName', 'surfCoords_mm_X', 'surfCoords_mm_Y',
       'Aquila_surfCoords_um_X', 'Aquila_surfCoords_um_Y',
       'Aquila_distance_um', 'Aquila_angle_deg', 'defect_Z_global_mm',
       'defectSurfDiameter_um', 'defectSurfMajorAxisLength_um',
       'defectSurfMinorAxisLength_um', 'defectSurfArea_um2', 'rawImgDownSamp']
'''
my_cols = ['blemishID', 'lensZoneID', 'lensZonePass', 'displayZoneID', 'dispZonePass', 'cleanable']

df_zone = df[my_cols]
df_fails = df_zone[(df_zone['lensZonePass'] == 0) | (df['dispZonePass'] == 0)]
print(df_fails)
