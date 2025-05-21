"""
@author: Nikunj K. Mangukiya (nikk.mangukiya@gmail.com)
"""

""" User Inputs """
data_folder_path = r'data_folder_path'  # File path where CAMELS_IND_dataset is extracted from zip file
# Example: data_folder_path = r'E:\CAMELS_IND_All_Catchments'
out_folder_path = r'out_folder_path'  # File paths where you want to save the filtered catchment data
# Example: out_folder_path = r'E:\CAMELS_IND_Catchments_Streamflow_Sufficient'

# provide required minimum percentage of flow data availability
data_percentage_required = 30

# shapefile clipping options:
shapefile_opt = 1
# 0: if you don't want to clip shapefiles
# 1: if you want to clip the shapefiles (will require geopandas package [https://pypi.org/project/geopandas/])

"""Following script will automatically filter out subset of dataset in provided out_folder_path based on User Inputs"""
# Import necessary packages
import os
import pandas as pd

# Filtering gauge ids
gauge_name = pd.read_csv(os.path.join(data_folder_path, 'attributes_txt', 'camels_ind_name.txt'), delimiter=';')
filt_gauge_name = gauge_name[gauge_name['flow_availability'] >= data_percentage_required]
filt_gauge_ids = filt_gauge_name['gauge_id'].reset_index(drop=True)
print(f"Preparing a subset of {len(filt_gauge_ids)} catchments with at least {data_percentage_required}% of flow data.")

# Filtering attributes
attr_lst = ['name', 'topo', 'clim', 'hydro', 'land', 'soil', 'geol', 'anth']
os.makedirs(os.path.join(out_folder_path, 'attributes_csv'), exist_ok=True)
for attr in attr_lst:
    file_path = os.path.join(data_folder_path, 'attributes_csv', f'camels_ind_{attr}.csv')
    df = pd.read_csv(file_path)
    filt_df = df[df['gauge_id'].isin(filt_gauge_ids)]
    output_file_path = os.path.join(out_folder_path, 'attributes_csv', f'camels_ind_{attr}.csv')
    filt_df.to_csv(output_file_path, index=False)
print("Filtered attributes_csv")
os.makedirs(os.path.join(out_folder_path, 'attributes_txt'), exist_ok=True)
for attr in attr_lst:
    file_path = os.path.join(data_folder_path, 'attributes_txt', f'camels_ind_{attr}.txt')
    df = pd.read_csv(file_path, delimiter=';')
    filt_df = df[df['gauge_id'].isin(filt_gauge_ids)]
    output_file_path = os.path.join(out_folder_path, 'attributes_txt', f'camels_ind_{attr}.txt')
    filt_df.to_csv(output_file_path, sep=';', index=False)
print("Filtered attributes_txt")

# Filtering catchment mean forcings
os.makedirs(os.path.join(out_folder_path, 'catchment_mean_forcings'), exist_ok=True)
for i in range(len(filt_gauge_ids)):
    id = filt_gauge_ids[i]
    df = pd.read_csv(os.path.join(data_folder_path, 'catchment_mean_forcings', f"{id:05d}.csv"))
    df.to_csv(os.path.join(out_folder_path, 'catchment_mean_forcings', f"{id:05d}.csv"), index=False)
print("Filtered catchment_mean_forcings")

# Filtering streamflow time series
os.makedirs(os.path.join(out_folder_path, 'streamflow_timeseries'), exist_ok=True)
df = pd.read_csv(os.path.join(data_folder_path, 'streamflow_timeseries', "streamflow_observed.csv"))
df = df.set_index(['year', 'month', 'day'])
df = df[df.columns.intersection(map(str, filt_gauge_ids))]
df = df.reset_index()
df.to_csv(os.path.join(out_folder_path, 'streamflow_timeseries', "streamflow_observed.csv"), index=None)
df = pd.read_csv(os.path.join(data_folder_path, 'streamflow_timeseries', "lstm_pred_streamflow.csv"))
df = df.set_index(['year', 'month', 'day'])
df = df[df.columns.intersection(map(str, filt_gauge_ids))]
df = df.reset_index()
df.to_csv(os.path.join(out_folder_path, 'streamflow_timeseries', "lstm_pred_streamflow.csv"), index=None)
print("Filtered streamflow_timeseries")

# Filtering shapefiles
if shapefile_opt == 1:
    import geopandas as gpd
    os.makedirs(os.path.join(out_folder_path, 'shapefiles_catchment'), exist_ok=True)
    catchment_shapefile = os.path.join(data_folder_path, 'shapefiles_catchment', 'merged', 'all_catchments.shp')
    gauge_shapefile = os.path.join(data_folder_path, 'shapefiles_catchment', 'merged', 'all_gauge_stations.shp')
    catchment_gdf = gpd.read_file(catchment_shapefile)
    gauge_gdf = gpd.read_file(gauge_shapefile)
    ids_5digit = [str(gauge_id).zfill(5) for gauge_id in filt_gauge_ids]
    filtered_catchments = catchment_gdf[catchment_gdf['gauge_id'].astype(str).isin(ids_5digit)]
    filtered_gauges = gauge_gdf[gauge_gdf['gauge_id'].astype(str).isin(ids_5digit)]
    filtered_catchments.to_file(os.path.join(out_folder_path, 'shapefiles_catchment', 'catchments.shp'))
    filtered_gauges.to_file(os.path.join(out_folder_path, 'shapefiles_catchment', 'gauge_stations.shp'))
    print("Clipped shapefiles")
else:
    print("Opted No for clipping shapefiles")

print("Data filtering completed...")
