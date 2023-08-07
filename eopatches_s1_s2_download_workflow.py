# Sentinel-1 and Sentinel-2 EOPatches for Maya architecture - AiTLAS competition
#
# Preparation of EOPatches with the following procedure:
# Part 1:
# Request Sentinel-2 (S2) data from Sentinel Hub for a set of areas and dates,
# creating an EOPatch for each area and filling EOPatch with S2 imagery for selected dates, then
# save S2 EOPatches to disk.
# Lists of dates and areas are imported from .txt files.
# Dates of S2 imagery were pre-selected: only dates with cloudless images are used here.
# Input .txt files structure:
# - list of dates for S2 data (.txt file with dates):
#     yyyy-mm-dd
#     yyyy-mm-dd
#     ...
#
# - list of areas (ids and coordinates of bboxes):
#     id xmin ymin xmax ymax
#     id xmin ymin xmax ymax
#     ...
#
# Part 2:
# Request Sentinel-1 (S1) data from Sentinel Hub (all images in time interval 2017-2019),
# add S1 imagery to EOPatch, calculate some S1 statistics and save S1 EOPatches to disk.
#
# Part 3:
# Add S1 data (statistics) to S2 EOPatches and save updated EOPatches to separate location on disk.
#
# A. Draksler, nov. 2020
# M. Somrak, apr. 2023

import os
import numpy as np
import time
from eolearn.core import EOWorkflow, FeatureType, OverwritePermission, EOTask, SaveTask, \
    EOExecutor, CreateEOPatchTask, MoveFeatureTask, EONode, linearly_connect_tasks
from eolearn.io import SentinelHubInputTask, SentinelHubEvalscriptTask
from eolearn.mask import CloudMaskTask
from sentinelhub import BBox, CRS, DataCollection, SHConfig
import datetime
# import warnings

global add_S1_ASC_data
global add_S1_DES_data
global stats_ASC
global stats_DES
global save_S1_ASC
global save_S1_DES

global list_of_dates
global add_S2_data
global S2_custom_CLM
global add_clm
global valid_mask
global save_S2

global move_S1_ASC_to_S2
global move_S1_DES_to_S2
global save_S1_S2

global eopatches_S2_folder
global eopatches_S1_ASC_folder
global eopatches_S1_DES_folder
global eopatches_folder
global reports_folder


'''
==================================================================================
Configuration and processing parameters
==================================================================================
'''

def parse_variables_from_txt(credentials_file, variables_list):
    # Read from file if it exist, otherwise set manually
    with open(credentials_file, 'r') as file:
        for line in file:
            name, value = line.split('=')
            name = name.strip()
            value = value.strip().strip("'")

            if name in variables_list:
                globals()[name] = value

def sentinel_hub_config(credentials_file) -> SHConfig():
    """ SentinelHub set configuration from credentials file """
    config = SHConfig()

    credentials_list = ['INSTANCE_ID', 'CLIENT_ID', 'CLIENT_SECRET']
    parse_variables_from_txt(credentials_file, credentials_list)

    config.instance_id = INSTANCE_ID
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET

    print("Instance ID: " + config.instance_id)
    return config



'''
==================================================================================
Processing parameters
==================================================================================
'''

def get_evalscript(band_name):
    """
    Evalscript for S1
    """

    evalscript = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ["VV", "VH"]
                    }],
                    output: [
                        { id:"<band_name>", bands:2, sampleType: SampleType.FLOAT32 }
                    ]
                }
            }

            function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
                outputMetadata.userData = { "norm_factor":  inputMetadata.normalizationFactor }
            }

            // apply "toDb" function on input bands
             function evaluatePixel(samples) {
              var VVdB = toDb(samples.VV)
              var VHdB = toDb(samples.VH)
              return [VVdB, VHdB]
            }

            // definition of "toDb" function
            function toDb(linear) {
              var log = 10 * Math.log(linear) / Math.LN10   // VV and VH linear to decibels
              var val = Math.max(Math.min(log, 5), -30) 
              return val
            }    
            """

    return evalscript.replace('<band_name>', band_name)

def process_S1(res,
               config,
               eopatches_S1_ASC_folder,
               eopatches_S1_DES_folder,
               time_interval,
               S1_stats=['mean', 'median', 'std', 'var', 'percentile'],  # statistics to calculate;
               # options are ['mean', 'median', 'std', 'var', 'percentile'],
               percentiles=[5, 95] # if you included 'percentile' is S1_stats, enter which percentiles to calculate
               ):
    """
    Process Sentinel 1 data
    """

    # request for S1 data
    """
    NOTE:
    We request Sentinel-1 data through SentinelHubEvalscriptTask with the following parameters:
        - Product type: level-1 GRD
        - Acquisition mode: IW
        - Resolution: high
        - Polarization: dual, VV+VH
        - Orbit direction: ascending, descending
        - Backscatter coefficient: Sigma0
    Values (linear power) of backscatter coefficient are converted to decibels (to get better distribution),
    then values are fitted to [-30, 5] dB interval (<-30 to -30 dB, >5 to 5 dB) and normalized to [0,1] interval.
    Interval [-30,5] dB is used because vast majority of values lies within in fact, >95% are smaller than 0 dB. Some 
    artificial surfaces and volumes can have extremely high values, above 5 dB. 
    More info:
    https://forum.step.esa.int/t/classification-sentinel-1-problems-with-maxver/12627/2
    https://docs.sentinel-hub.com/api/latest/data/sentinel-1-grd/#units
    """

    global add_S1_ASC_data
    global add_S1_DES_data
    global stats_ASC
    global stats_DES
    global save_S1_ASC
    global save_S1_DES

    # S1 ASC data:

    add_S1_ASC_data = SentinelHubEvalscriptTask(
        evalscript=get_evalscript('BANDS_S1_ASC'),
        data_collection=DataCollection.SENTINEL1_IW_ASC,
        features=(FeatureType.DATA, 'BANDS_S1_ASC'),
        resolution=res,
        time_difference=datetime.timedelta(minutes=120),
        config=config,
        aux_request_args={"processing": {"backCoeff": "SIGMA0_ELLIPSOID"}},
    )

    # S1 DES data:

    add_S1_DES_data = SentinelHubEvalscriptTask(
        evalscript=get_evalscript('BANDS_S1_DES'),
        data_collection=DataCollection.SENTINEL1_IW_DES,
        features=(FeatureType.DATA, 'BANDS_S1_DES'),
        resolution=res,
        time_difference=datetime.timedelta(minutes=120),
        config=config,
        aux_request_args={"processing": {"backCoeff": "SIGMA0_ELLIPSOID"}},
    )

    # create output_features for MapFeatureYearlyTask
    output_ASC_features = []
    output_DES_features = []
    S1_stats.sort(key='percentile'.__eq__)
    for s in S1_stats:
        if s == 'percentile':
            s = 'p'
            for p in percentiles:
                output_ASC_features.append((FeatureType.DATA_TIMELESS, 'S1_ASC_{}{}_'.format(s, p)))
                output_DES_features.append((FeatureType.DATA_TIMELESS, 'S1_DES_{}{}_'.format(s, p)))
        else:
            output_ASC_features.append((FeatureType.DATA_TIMELESS, 'S1_ASC_{}_'.format(s)))
            output_DES_features.append((FeatureType.DATA_TIMELESS, 'S1_DES_{}_'.format(s)))

    # calculation of statistics for VV and VH bands (ASC and DES)
    stats_ASC = MapFeatureYearlyTask(input_feature=(FeatureType.DATA, 'BANDS_S1_ASC'),
                                     output_features=output_ASC_features,
                                     map_functions=S1_stats,
                                     percentiles=percentiles,
                                     axis=0)
    stats_DES = MapFeatureYearlyTask(input_feature=(FeatureType.DATA, 'BANDS_S1_DES'),
                                     output_features=output_DES_features,
                                     map_functions=S1_stats,
                                     percentiles=percentiles,
                                     axis=0)

    # save S1 EOPatch
    save_S1_ASC = RetrySaveTask(eopatches_S1_ASC_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    save_S1_DES = RetrySaveTask(eopatches_S1_DES_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    return


def process_S2(res, # resolution
               S2_cloudless_dates_file,
               eopatches_S2_folder, # output folder
               S2_cloudless_dates=True, # have you pre-selected cloudless dates with Sentinel-2 images for your area? If not, False
               selected_max_cc=0.8,  # maximum cloud cover on requested S2 data,
               S2_band_names=['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'],
               calculate_S2_custom_CLM=True,  # If True, additional cloud mask (CLM) will be calculated. If False, Sentinel Hub CLM is used
               # S2 cloud mask calculation parameters (Only used if calculate_S2_custom_CLM = True)
               clm_res=10, # resolution for cloud mask calculation in meters
               clm_average_over=22, # recommended value for 10m res. Size of the pixel neighbourhood used
               # in the averaging post-processing step.
               clm_dilation_size=11  # recommended value for 10m res. Size of the dilation post-processing step.
               ):
    """
    Process Sentinel 2 data
    """

    global add_S2_data
    global S2_custom_CLM
    global add_clm
    global valid_mask
    global save_S2
    global list_of_dates

    data_collection_S2 = DataCollection.SENTINEL2_L2A

    # import dates for S2 imagery
    list_of_dates = []
    with open(S2_cloudless_dates_file, mode='r') as infile_dates:
        for line in infile_dates.read().splitlines():
            current_date = datetime.datetime.strptime(line, "%Y-%m-%d")
            list_of_dates.append(current_date)
    list_of_dates.sort()

    # request for S2 data (bands + cloud masks)
    add_S2_data = SentinelHubInputTask(
        bands_feature=(FeatureType.DATA, 'BANDS_S2'),
        bands=S2_band_names,
        resolution=res,
        maxcc=selected_max_cc,
        data_collection=data_collection_S2,
        time_difference=datetime.timedelta(minutes=120),
        additional_data=[(FeatureType.MASK, 'CLM'),  # SH cloud mask (res. 160m)
                         (FeatureType.MASK, 'dataMask', 'IS_DATA')],
        config=config)

    # calculate your own cloud mask
    S2_custom_CLM = CloudMaskTask(data_feature=(FeatureType.DATA, 'BANDS_S2'),
                                  all_bands=False,  # all 13 bands or only the required 10
                                  processing_resolution=clm_res,
                                  mono_features=(None, 'CLM_{}m'.format(clm_res)),  # names of output features
                                  mask_feature=None,
                                  average_over=clm_average_over,
                                  dilation_size=clm_dilation_size)

    add_clm = CloudMaskTask(data_feature=(FeatureType.DATA, 'BANDS_S2'),
                            all_bands=True,
                            processing_resolution=160,
                            mono_features=('CLP', 'CLM'),
                            mask_feature=None,
                            average_over=16,
                            dilation_size=8)

    # add "valid data" feature
    CLM = 'CLM_{}m'.format(clm_res) if calculate_S2_custom_CLM else 'CLM'
    # VALIDITY MASK
    # Validate pixels using SentinelHub's cloud detection mask and region of acquisition
    valid_mask = SentinelHubValidDataTask((FeatureType.MASK, "IS_VALID"), cloud_mask=CLM, data_mask='IS_DATA')

    # save S2 EOPatch
    save_S2 = RetrySaveTask(eopatches_S2_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

def merge_S1_S2():
    global move_S1_ASC_to_S2
    global move_S1_DES_to_S2
    global save_S1_S2

    # move S1 statistics to S2 EOPatch
    move_S1_ASC_to_S2 = MoveFeatureTask(FeatureType.DATA_TIMELESS)
    move_S1_DES_to_S2 = MoveFeatureTask(FeatureType.DATA_TIMELESS)

    # save updated EOPatch with S2 and S1 data
    save_S1_S2 = RetrySaveTask(path=eopatches_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

def set_output_locations(output_locations_file):
    global eopatches_S2_folder
    global eopatches_S1_ASC_folder
    global eopatches_S1_DES_folder
    global eopatches_folder
    global reports_folder

    credentials_list = ['eopatches_S2_folder',
                        'eopatches_S1_ASC_folder',
                        'eopatches_S1_DES_folder',
                        'eopatches_folder',
                        'reports_folder']

    parse_variables_from_txt(output_locations_file, credentials_list)

    out_folders = (eopatches_S2_folder,
                   eopatches_S1_ASC_folder,
                   eopatches_S1_DES_folder,
                   eopatches_folder,
                   reports_folder)

    for f in out_folders:
        if not os.path.isdir(f):
            os.makedirs(f)

def execute_workflow(workflow, execution_args, reports_folder):
    # TODO: SHDeprecationWarning: The string representation of `BBox` will change to match its `repr` representation value = format % v
    print('Now creating {} EOPatches...\n'.format(len(list_of_bboxes)))
    executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder=reports_folder)
    executor.run(workers=1, multiprocess=False)
    executor.make_report()

    print(f"Report was saved to location: {executor.get_report_path()}")

def merge(dict1, dict2):
    """ Merge entries from two dictionaries into one. """
    res = {**dict1, **dict2}
    return res

'''
==================================================================================
Define custom EOTasks
==================================================================================
'''

# define EOTask for S1 yearly means
class MapFeatureYearlyTask(EOTask):
    """
    Applies one or more functions to a feature and feature's yearly subset in input_feature of EOPatch
    and stores the results in a set of output_features. Modified from MapFeatureTask.
    """

    def __init__(self, input_feature, output_features, aggregate="year", map_functions=None, percentiles=None, **kwargs):
        """
        :param input_features: Input feature to be mapped. Only one feature at the moment.
        :type input_feature: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param output_features: A collection of the output features to which to assign the output data.
        :type output_features: an object supported by the :class:`FeatureParser<eolearn.core.utilities.FeatureParser>`
        :param aggregate: aggregate of input_features to which functions will be passed. Only "year" is implemented
        at the moment.
        :param map_function: A function or list of functions to be applied to the input data.
        :raises ValueError: Raises an exception when number of functions and number of output_features do not match.
        :param kwargs: kwargs to be passed to the map function.
        """

        self.input_features = list(self.parse_feature(input_feature))
        self.output_features = list(self.parse_features(output_features))
        self.agg = aggregate
        if self.agg != 'year':
            raise ValueError('Only "year" is implemented at the moment.')
        self.kwargs = kwargs

        if 'percentile' in map_functions:
            map_functions = [i for i in map_functions if i != 'percentile']
            m = ['percentile'] * len(percentiles)
            [map_functions.append(i) for i in m]

            perc1 = [''] * (len(map_functions) - len(percentiles))
            [perc1.append(i) for i in percentiles]
            self.percentiles = perc1

            self.kwargs_list = []
            for x in range(len(map_functions)):
                kwargs_dict = {}
                for k, v in self.kwargs.items():
                    kwargs_dict[k] = v
                self.kwargs_list.append(kwargs_dict)
            for mf, p, kwl in zip(map_functions, self.percentiles, self.kwargs_list):
                if mf == 'percentile':
                    kwl['q'] = p
        else:
            self.kwargs_list = []
            for x in range(len(map_functions)):
                kwargs_dict = {}
                for k, v in self.kwargs.items():
                    kwargs_dict[k] = v
                self.kwargs_list.append(kwargs_dict)

        self.functions = [getattr(np, i) for i in map_functions]


    def execute(self, eopatch):
        """
        :param eopatch: Source EOPatch from which to read the data of input features.
        :type eopatch: EOPatch
        :return: An eopatch with the additional mapped features.
        :rtype: EOPatch
        """

        # Extract indexes of first and last timestamp for every year.
        yrs_list = [getattr(i, self.agg) for i in eopatch.timestamps]  # get year from datetime.datetime objects
        yrs_uniq = list(set(yrs_list))  # get unique values of years
        yrs_uniq.sort()
        yrs_start_end = []
        for yr in yrs_uniq:
            yr_indexes = [i for i in range(len(yrs_list)) if yrs_list[i] == yr]  # indexes of each year
            yrs_start_end.append((yr_indexes[0], yr_indexes[-1]))  # append tuple of first and last idx to list

        # pass functions to features
        for input_features in self.input_features:
            for output_feature, fun, kwgs in zip(self.output_features, self.functions, self.kwargs_list):

                # add period suffix to feature name
                output_feature_l = list(output_feature)
                # Old naming:
                #output_feature_l[1] = '{}{}-{}'.format(output_feature_l[0], yrs_uniq[0], yrs_uniq[-1])
                # New naming:
                output_feature_l[1] = '{}-{}'.format(yrs_uniq[0], yrs_uniq[-1])
                output_feature_id = tuple(output_feature_l)

                # pass function to input feature
                if 'BANDS_S1_ASC' in eopatch[FeatureType.DATA].keys():
                    bands_s1 = eopatch[FeatureType.DATA, 'BANDS_S1_ASC']
                elif 'BANDS_S1_DES' in eopatch[FeatureType.DATA].keys():
                    bands_s1 = eopatch[FeatureType.DATA, 'BANDS_S1_DES']
                else:
                    bands_s1 = None
                eopatch[output_feature_id] = fun(bands_s1, **kwgs)
                # eopatch[output_feature_id] = fun(eopatch[FeatureType.DATA], **kwgs)

                for yr_suffix, [start_idx, end_idx] in zip(yrs_uniq, yrs_start_end):

                    # add year suffix to feature name
                    output_feature_y_l = list(output_feature)
                    output_feature_y_l[1] = output_feature_y_l[1] + str(yr_suffix)
                    output_feature_y_id = tuple(output_feature_y_l)

                    # pass function to subset of input feature
                    eopatch[output_feature_y_id] = fun(bands_s1[start_idx:end_idx + 1], **kwgs)
                    # eopatch[output_feature_y_id] = fun(eopatch[FeatureType.DATA][start_idx:end_idx + 1], **kwgs)

        return eopatch


class SentinelHubValidDataTask(EOTask):
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """

    def __init__(self, output_feature, cloud_mask='CLM', data_mask='IS_DATA'):
        self.output_feature = output_feature
        self.cloud_mask = cloud_mask
        self.data_mask = data_mask

    def execute(self, eopatch):
        # Ensure both masks are boolean
        data_mask_bool = eopatch.mask[self.data_mask].astype(bool)
        cloud_mask_bool = eopatch.mask[self.cloud_mask].astype(bool)

        # Combine masks: valid where data is present and no clouds
        eopatch[self.output_feature] = data_mask_bool & (~cloud_mask_bool)
        return eopatch


class RetrySaveTask(SaveTask):
    def __init__(self, path, retries=100, delay=2, **kwargs):
        super().__init__(path, **kwargs)
        self.retries = retries
        self.delay = delay

    def execute(self, eopatch, **kwargs):
        for i in range(self.retries):
            try:
                return super().execute(eopatch, **kwargs)
            except OSError as e:
                if "[WinError 6]" in str(e) and i < self.retries - 1:  # Handle is invalid
                    time.sleep(self.delay) # retry in case of concurrent access
                else:
                    raise  # Re-raise the last exception


'''
==================================================================================
Main program - data requests, processing and saving
==================================================================================
'''
if __name__ == '__main__':
    # start the clock
    time_start = datetime.datetime.now()

    S2_cloudless_dates_file = './input_files/valid_dates_S2.txt'  # file with list of S-2 cloudless dates (optional)
    bboxes_file = './input_files/list_of_bboxes.txt'  # file with bounding boxes

    credentials_file = "./user_credentials.txt"
    config = sentinel_hub_config(credentials_file)
    output_locations_file = "./set_output_folders.txt"
    set_output_locations(output_locations_file)

    list_of_bboxes = []
    with open(bboxes_file, mode='r') as infile_bboxes:
        for line in infile_bboxes.readlines():
            line = line.split(sep=' ')
            id, xmin, ymin, xmax, ymax = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
            coords = (xmin, ymin, xmax, ymax)
            list_of_bboxes.append([id, coords])

    """
    ===============================================================
    EOTasks for creating EOPatches, adding S2 and S1 data, S2 and S1 processing,
    saving to disk, merging S1 and S2 data 
    ===============================================================
    """

    # parameters
    crs = CRS.UTM_16N  # CRS of bboxes - for Maya sites in Chactun: EPSG:32616 - WGS 84 / UTM zone 16N
    res = 10  # resolution of EOPatches in meters

    sentinel1 = True
    sentinel2 = True

    # request for Sentinel 2 data
    if sentinel2:
        process_S2(res, S2_cloudless_dates_file, eopatches_S2_folder)

    # request for Sentinel) 1 data
    if sentinel1:
        # S1 processing parameters
        time_interval = ['2017-01-01', '2018-12-31']
        # time_interval = ['2017-01-01', '2020-12-31']  # time interval for S1 in format ['YYYY-MM-DD', 'YYYY-MM-DD']
        process_S1(res, config, eopatches_S1_ASC_folder, eopatches_S1_DES_folder, time_interval)

    #
    if sentinel1 and sentinel2:
        merge_S1_S2()


    """
    =================================================================================
    Creating EOWorkflow, then using EOExecutor to loop through areas (all EONodes)
    =================================================================================
    """

    create_eopatch = CreateEOPatchTask()
    # New workflow with nodes, EONodes
    nodes = []

    # Sentinel 2 tasks:
    if sentinel2:
        node1 = EONode(create_eopatch, [], 'Create S2 EOPatch')
        node2 = EONode(add_S2_data, [node1], 'Add S2 data')
        node3 = EONode(S2_custom_CLM, [node2], 'Add your own cloud mask')
        node4 = EONode(valid_mask, [node3], 'Add "valid data" mask')
        node5 = EONode(save_S2, [node4], 'Save S2 EOPatch')
        nodes_S2 = [node1, node2, node3, node4, node5]

    # Sentinel 1 tasks:
    if sentinel1:
        node6 = EONode(add_S1_ASC_data, [], 'Add S1_ASC data')
        node7 = EONode(stats_ASC, [node6], 'Stats S1_ASC')
        node8 = EONode(save_S1_ASC, [node7], 'Save S1_ASC EOPatch')
        node9 = EONode(add_S1_DES_data, [], 'Add S1_DES data')
        node10 = EONode(stats_DES, [node9], 'Stats S1_DES')
        node11 = EONode(save_S1_DES, [node10], 'Save S1_DES EOPatch')
        nodes_S1 = [node6, node7, node8, node9, node10]

    # Sentinel 1 and 2 join stats:
    if sentinel1 and sentinel2:
        node12 = EONode(move_S1_ASC_to_S2, [node7, node4], 'Move S1_ASC stats to S2 EOPatch')
        node13 = EONode(move_S1_DES_to_S2, [node10, node12], 'Move S1_DES stats to S2 EOPatch')
        node14 = EONode(save_S1_S2, [node13], 'Save EOPatch S1 and S2')
        nodes_S1_S2 = [node11, node12, node13, node14]


    # Workflow:
    workflow_nodes = []
    if sentinel1: # S1 SAR
        workflow_nodes = workflow_nodes + nodes_S1
    if sentinel2: # S2 optical
        workflow_nodes = workflow_nodes + nodes_S2
        if sentinel1: # merge S1 abd S2
            workflow_nodes = workflow_nodes + nodes_S1_S2

    workflow = EOWorkflow(workflow_nodes)

    # Execution arguments:
    execution_args = []
    for bbox_id, bbox_coords in list_of_bboxes:
        dict_args = {}

        if sentinel1: # S1 SAR
            dict_sentinel1 = {
                node6: {'bbox': BBox(bbox_coords, crs=crs), 'time_interval': time_interval},
                node8: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)},
                node9: {'bbox': BBox(bbox_coords, crs=crs), 'time_interval': time_interval},
                node11: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)}
            }
            dict_args = merge(dict_args, dict_sentinel1)

        if sentinel2: # S2 optical
            dict_sentinel2 = {
                node1: {'bbox': BBox(bbox_coords, crs=crs), 'timestamps': list_of_dates},
                node2: {'bbox': BBox(bbox_coords, crs=crs)},
                node5: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)}
            }
            dict_args = merge(dict_args, dict_sentinel2)

            if sentinel1: # for merging S1 and S2
                dict_merge = {
                    node14: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)}
                }
                dict_args = merge(dict_args, dict_merge)

        execution_args.append(dict_args)

    '''
    ==================== end eoworlflow =======================================================
    '''

    execute_workflow(workflow, execution_args, reports_folder)

    time_end = datetime.datetime.now()
    time_duration = time_end - time_start
    print('\nExecution finished in: ', time_duration)

