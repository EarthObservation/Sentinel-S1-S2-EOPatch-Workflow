# Export from EOpatch to Tiff

import os
import re
import numpy as np
from eolearn.io import ExportToTiff
from eolearn.core import EOPatch, FeatureType, InitializeFeature
import datetime
from tqdm import tqdm

# folder with input eopatches
eopatches_folder =  '' # folder as string

# Sentinel-1 and/or Sentinel-2 export
selection = ['S2']  # select which satellite data to export (insert 'S1' and/or 'S2'; e.g. ['S1', 'S2']

# output folder (subfolders 'S1/S2' will be auto-created)
tiff_folder = './images/TIFF'  # local

# parameters from S1 statistics
years =  ['2017', '2018', '2019', '2017-2019'] # years as list of strings or span of years
orbits = ['ASC', 'DES']
polarizations = ['VV', 'VH']
stats = ['mean', 'median', 'std', 'var', 'p5', 'p95']

if __name__ == '__main__':
    time_start = datetime.datetime.now()

    # create output folders
    if 'S1' in selection or 'S2' in selection:
        for s in selection:
            tiff_subfolder = os.path.join(tiff_folder, s)
            if not os.path.isdir(tiff_subfolder):
                os.makedirs(tiff_subfolder)

    """
    ====================================
    Sentinel-1 statistics to Tiff files
    ====================================
    """
    if 'S1' in selection:
        # create a list of S1 statistics inside eopatch that will represent bands of output Tiff files
        list_of_stats = []
        for year in years:
            for orbit in orbits:
                for polar in polarizations:
                    for stat in stats:
                        combination = '{}_{}_{}_{}'.format(year, orbit, polar, stat)
                        naming = 'S1_{}_{}_{}'.format(orbit, stat, year)
                        pair = dict(combination=combination, naming=naming)
                        list_of_stats.append(pair)

        eopatches = os.listdir(eopatches_folder)

        # loop through eopatches
        for eopatch_id in tqdm(range(len(eopatches))):
            eopatch = EOPatch.load(os.path.join(eopatches_folder, 'eopatch_{}'.format(eopatch_id)), lazy_loading=True)

            # crete temporary feature for all statistics
            new_feature = InitializeFeature((FeatureType.DATA_TIMELESS, 'S1_all_stats'),
                                            shape=(24, 24, len(list_of_stats)),
                                            dtype=np.float64)
            new_feature.execute(eopatch)

            # extract all statistics to new_feature
            for sid, stat_dict in enumerate(list_of_stats):

                polar_id = 0 if 'VV' in stat_dict['combination'] else 1
                # print(sid, stat_dict['naming'], ' | ', stat_dict['combination'], ' | ', 'polar_id:', polar_id)
                eopatch.data_timeless['S1_all_stats'][..., sid] = eopatch.data_timeless[stat_dict['naming']][..., polar_id]

            # export new_feature to Tiff
            export = ExportToTiff((FeatureType.DATA_TIMELESS, 'S1_all_stats'), folder=tiff_subfolder)
            export.execute(eopatch, filename='tile_{}.tiff'.format(eopatch_id))

    """
    ====================================
    Sentinel-2 data to Tiff files
    ====================================
    """
    if 'S2' in selection:
        eopatches = os.listdir(eopatches_folder)

        # loop through eopatches
        for eopatch_id in tqdm(range(len(eopatches))):
            eopatch = EOPatch.load(os.path.join(eopatches_folder, 'eopatch_{}'.format(eopatch_id)), lazy_loading=True)

            # crete temporary feature for all S2 bands (~12 bands + CLM)
            new_feature = InitializeFeature((FeatureType.DATA, 'S2_bands_and_clm'),
                                            shape=(eopatch.data['BANDS_S2'].shape[0],
                                                   eopatch.data['BANDS_S2'].shape[1],
                                                   eopatch.data['BANDS_S2'].shape[2],
                                                   eopatch.data['BANDS_S2'].shape[3]+1),
                                            dtype=np.float32)
            new_feature.execute(eopatch)

            # extract all S2 bands and CLM to new_feature
            eopatch.data['S2_bands_and_clm'][..., :-1] = eopatch.data['BANDS_S2']

            if any(re.search(r'CLM_\d+m', k) for k in eopatch.mask.keys()):
                for key in eopatch.mask.keys():
                    if re.search(r'CLM_\d+m', key):
                        eopatch.data['S2_bands_and_clm'][..., -1:] = eopatch.mask[key]  # 'e.g. CLM_10m'
 
            else:
                eopatch.data['S2_bands_and_clm'][..., -1:] = eopatch.mask['CLM']

            # export new_feature to Tiff
            export = ExportToTiff((FeatureType.DATA, 'S2_bands_and_clm'), folder=tiff_subfolder)
            export.execute(eopatch, filename='tile_{}.tiff'.format(eopatch_id))

    else:
        raise NameError("'S1' and/or 'S2' expected in 'selection' variable, got {} instead".format(selection))

    time_end = datetime.datetime.now()
    time_duration = time_end - time_start
    print('\nExecution finished in: ', time_duration)
