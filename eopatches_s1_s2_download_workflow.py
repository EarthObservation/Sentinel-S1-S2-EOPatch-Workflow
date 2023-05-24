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
# M. somrak, apr. 2023


import os
import numpy as np

# TODO: is deprecated; to replace
# EODeprecationWarning: The `FeatureTypeSet` collections are deprecated.
# The argument `allowed_feature_types` of feature parsers can now be a callable,
# so you can use `lambda ftype: ftype.is_spatial()` instead of `FeatureTypeSet.SPATIAL_TYPES` in such cases.
#   xml += var_to_xml(v, str(k), evaluate_full_value=eval_full_val)

from eolearn.core import EOWorkflow, FeatureType, OverwritePermission, EOTask, SaveTask, \
    EOExecutor, CreateEOPatchTask, MoveFeatureTask, EONode, linearly_connect_tasks
from eolearn.io import SentinelHubInputTask
from eolearn.mask import CloudMaskTask
from sentinelhub import BBox, CRS, DataCollection, SHConfig
import datetime
# import warnings

'''
==================================================================================
Configuration and processing parameters
==================================================================================
'''
# Set INSTANCE_ID from configuration in SentinelHub account
INSTANCE_ID = ''  # SH configuration: AD-AiTLAS-config, string parameter
CLIENT_ID = ''  # if already in SH config file (config.json) ($ sentinelhub.config --show), leave it as is
CLIENT_SECRET = ''  # if already in SH config file (config.json) ($ sentinelhub.config --show), leave it as is

# Can also be set using:
# $sentinelhub.config --instance_id "$INSTANCE_ID" --sh_client_id "$CLIENT_ID" --sh_client_secret "$CLIENT_SECRET"
# or
# $ sentinelhub.config --profile my-profile --instance_id my-instance-id
# Confirm the values have been set:
# $sentinelhub.config --show

if INSTANCE_ID and CLIENT_ID and CLIENT_SECRET:
    config = SHConfig()
    config.instance_id = INSTANCE_ID
    config.sh_client_id = CLIENT_ID               # use only if you changed variable CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET       # use only if you changed variable CLIENT_SECRET
else:
    config = None

# CRS for Maya sites in Chactun: EPSG:32616 - WGS 84 / UTM zone 16N
# CRS of bboxes
crs = CRS.UTM_16N  # define CRS

# input files
S2_cloudless_dates = True  # have you pre-selected cloudless dates with Sentinel-2 images for your area? If not, False
S2_cloudless_dates_file = './input_files/valid_dates_S2.txt'  # file with list of S-2 cloudless dates (optional)
bboxes_file = './input_files/list_of_bboxes.txt'  # file with bounding boxes

# data collections from Sentinel Hub
data_collection_S1_ASC = DataCollection.SENTINEL1_IW_ASC
data_collection_S1_DES = DataCollection.SENTINEL1_IW_DES
data_collection_S2 = DataCollection.SENTINEL2_L2A

res = 10  # resolution of EOPatches in meters

# S1 processing parameters
time_interval = ['2017-01-01', '2020-12-31']  # time interval for S1 in format ['YYYY-MM-DD', 'YYYY-MM-DD']
S1_stats = ['mean', 'median', 'std', 'var', 'percentile']  # statistics to calculate;
# options are ['mean', 'median', 'std', 'var', 'percentile']
percentiles = [5, 95]  # if you included 'percentile' is S1_stats, enter which percentiles to calculate

# S2 processing parameters
selected_max_cc = 0.8  # maximum cloud cover on requested S2 data
S2_band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

# Use Sentinel Hub's default cloud mask (res. 160m) or calculate your own cloud mask?
calculate_S2_custom_CLM = True  # If True, additional cloud mask (CLM) will be calculated. If False, Sentinel Hub CLM is used

# S2 cloud mask calculation parameters (Only used if calculate_S2_custom_CLM = True)
clm_res = 10  # resolution for cloud mask calculation in meters
clm_average_over = 22  # recommended value for 10m res. Size of the pixel neighbourhood used
# in the averaging post-processing step.
clm_dilation_size = 11  # recommended value for 10m res. Size of the dilation post-processing step.


# output locations
eopatches_S2_folder = './eopatches_S2/'
eopatches_S1_ASC_folder = './eopatches_S1/ASC/'
eopatches_S1_DES_folder = './eopatches_S1/DES/'
eopatches_folder = './eopatches/'
reports_folder = './reports/'

out_folders = (eopatches_S2_folder,
               eopatches_S1_ASC_folder,
               eopatches_S1_DES_folder,
               eopatches_folder,
               reports_folder)
for f in out_folders:
    if not os.path.isdir(f):
        os.makedirs(f)

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

        # print(self.kwargs_list)

        self.functions = [getattr(np, i) for i in map_functions]

    def execute(self, eopatch):
        """
        :param eopatch: Source EOPatch from which to read the data of input features.
        :type eopatch: EOPatch
        :return: An eopatch with the additional mapped features.
        :rtype: EOPatch
        """

        # Extract indexes of first and last timestamp for every year.
        yrs_list = [getattr(i, self.agg) for i in eopatch.timestamp]  # get year from datetime.datetime objects
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
                output_feature_l[1] = '{}{}-{}'.format(output_feature_l[1], yrs_uniq[0], yrs_uniq[-1])
                output_feature_id = tuple(output_feature_l)

                # pass function to input feature
                eopatch[output_feature_id] = fun(eopatch[input_features], **kwgs)

                for yr_suffix, [start_idx, end_idx] in zip(yrs_uniq, yrs_start_end):

                    # add year suffix to feature name
                    output_feature_y_l = list(output_feature)
                    output_feature_y_l[1] = output_feature_y_l[1] + str(yr_suffix)
                    output_feature_y_id = tuple(output_feature_y_l)

                    # pass function to subset of input feature
                    eopatch[output_feature_y_id] = fun(eopatch[input_features][start_idx:end_idx + 1], **kwgs)

        return eopatch

class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA` mask
    Cloud mask is assumed to be found in eopatch.mask['CLM']
    """
    def __init__(self, data_mask, cloud_mask):
        """

        :param data_mask: name of EOPatch FeatureType.MASK with 'IS_DATA' information
        :type data_mask: string
        :param cloud_mask: name of EOPatch FeatureType.MASK with 'cloud mask' data
        :type cloud_mask:  string
        """
        self.data_mask = data_mask
        self.cloud_mask = cloud_mask

    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask[self.data_mask].astype(np.bool),  # 'IS_DATA'
                              np.logical_not(eopatch.mask[self.cloud_mask].astype(np.bool)))   # 'CLM' or 'CLM_10m'

# # define EOTask for adding valid data mask
# class SentinelHubValidData(EOTask):
#     """EOTask for adding custom mask array used to filter reflectances data
#     This task allows the user to specify the criteria used to generate a valid data mask, which can be used to
#     filter the data stored in the `FeatureType.DATA`
#     """
#
#     def __init__(self, predicate, valid_data_feature=(FeatureType.MASK, "VALID_DATA")):
#         """Constructor of the class requires a predicate defining the function used to generate the valid data mask. A
#         predicate is a function that returns the truth value of some condition.
#         An example predicate could be an `and` operator between a cloud mask and a snow mask.
#         :param predicate: Function used to generate a `valid_data` mask
#         :type predicate: func
#         :param valid_data_feature: Feature which will store valid data mask
#         :type valid_data_feature: str
#         """
#         self.predicate = predicate
#         self.valid_data_feature = self.parse_feature(valid_data_feature)
#
#     def execute(self, eopatch):
#         """Execute predicate on input eopatch
#         :param eopatch: Input `eopatch` instance
#         :return: The same `eopatch` instance with a `mask.valid_data` array computed according to the predicate
#         """
#         feature_type, feature_name = next(self.valid_data_feature())
#         eopatch[feature_type][feature_name] = self.predicate(eopatch)
#         return eopatch

class AddValidDataMaskTask(EOTask):
    """EOTask for adding custom mask array used to filter reflectances data
    This task allows the user to specify the criteria used to generate a valid data mask, which can be used to
    filter the data stored in the `FeatureType.DATA`
    """

    def __init__(self, predicate, valid_data_feature=(FeatureType.MASK, "VALID_DATA")):
        """Constructor of the class requires a predicate defining the function used to generate the valid data mask. A
        predicate is a function that returns the truth value of some condition.
        An example predicate could be an `and` operator between a cloud mask and a snow mask.
        :param predicate: Function used to generate a `valid_data` mask
        :type predicate: func
        :param valid_data_feature: Feature which will store valid data mask
        :type valid_data_feature: str
        """
        self.predicate = predicate
        self.valid_data_feature = self.parse_feature(valid_data_feature)
        return #tmp

    def execute(self, eopatch):
        """Execute predicate on input eopatch
        :param eopatch: Input `eopatch` instance
        :return: The same `eopatch` instance with a `mask.valid_data` array computed according to the predicate
        """
        feature_type, feature_name = next(self.valid_data_feature())
        eopatch[feature_type][feature_name] = self.predicate(eopatch)
        return eopatch

'''
==================================================================================
Main program - data requests, processing and saving
==================================================================================
'''
if __name__ == '__main__':
    # start the clock
    time_start = datetime.datetime.now()

    # import dates for S2 imagery
    list_of_dates = []
    with open(S2_cloudless_dates_file, mode='r') as infile_dates:
        for line in infile_dates.read().splitlines():
            current_date = datetime.datetime.strptime(line, "%Y-%m-%d")
            list_of_dates.append(current_date)
    list_of_dates.sort()

    list_of_bboxes = []
    with open(bboxes_file, mode='r') as infile_bboxes:
        for line in infile_bboxes.readlines():
            line = line.split(sep=' ')
            id, xmin, ymin, xmax, ymax = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
            coords = (xmin, ymin, xmax, ymax)
            list_of_bboxes.append([id, coords])

    # TODO: temporarily only use five samples, then change back
    list_of_bboxes = list_of_bboxes[:5]

    """
    ===============================================================
    EOTasks for creating EOPatches, adding S2 and S1 data, S2 and S1 processing,
    saving to disk, merging S1 and S2 data 
    ===============================================================
    """

    # create empty EOPatch
    # TODO: deprecation warning: need to init all EOPatches with bboxes
    create_eopatch = CreateEOPatchTask()

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

        config=config
    )

    # calculate your own cloud mask
    S2_custom_CLM = CloudMaskTask(data_feature=(FeatureType.DATA, 'BANDS_S2'),
                                          all_bands=False,  # all 13 bands or only the required 10
                                          processing_resolution=clm_res,
                                          mono_features=(None, 'CLM_{}m'.format(clm_res)),  # names of output features
                                          mask_feature=None,
                                          average_over=clm_average_over,
                                          dilation_size=clm_dilation_size)

    add_clm = CloudMaskTask(data_feature=(FeatureType.DATA, 'BANDS'),
                            all_bands=True,
                            processing_resolution=160,
                            mono_features=('CLP', 'CLM'),
                            mask_feature=None,
                            average_over=16,
                            dilation_size=8)

    #-----ORIGINAL-----
    # add "valid data" feature
    CLM = 'CLM_{}m'.format(clm_res) if calculate_S2_custom_CLM else 'CLM'
    # VALIDITY MASK
    # Validate pixels using SentinelHub's cloud detection mask and region of acquisition??
    # Before it was AddValidDataMaskTask, then was renamed to SentinelHubValidDataTask
    valid_mask = AddValidDataMaskTask(SentinelHubValidData(data_mask='IS_DATA',  # name of 'dataMask' feature
                                                           cloud_mask=CLM),  # name of 'cloud mask' feature
                                                           valid_data_feature=(FeatureType.MASK,'IS_VALID'))  # name of output mask

    # save S2 EOPatch
    save_S2 = SaveTask(eopatches_S2_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    
    # request for S1 data
    """
    NOTE:
    We request Sentinel-1 data through SentinelHubInputTask with the following parameters:
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

    evalscript = """
        //VERSION=3
    
    function setup() {
        return {
            input: [{
                bands: ["VV", "VH"]
            }],
            output: [
                { id:"custom", bands:2, sampleType: SampleType.FLOAT32 }
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

    # separate requests for S1 ascending and descending orbits
    add_S1_ASC_data = SentinelHubInputTask(evalscript=evalscript,
                                           data_collection=data_collection_S1_ASC,
                                           bands_feature=(FeatureType.DATA, 'BANDS_S1'),  # two bands inside: VV, VH
                                           resolution=res,
                                           time_difference=datetime.timedelta(minutes=120),
                                           config=config,
                                           aux_request_args={"processing": {"backCoeff": "SIGMA0_ELLIPSOID"}}
                                           )

    add_S1_DES_data = SentinelHubInputTask(evalscript=evalscript,
                                           data_collection=data_collection_S1_DES,
                                           bands_feature=(FeatureType.DATA, 'BANDS_S1'),
                                           resolution=res,
                                           time_difference=datetime.timedelta(minutes=120),
                                           config=config,
                                           aux_request_args={"processing": {"backCoeff": "SIGMA0_ELLIPSOID"}}
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
    stats_ASC = MapFeatureYearlyTask(input_feature=(FeatureType.DATA, 'BANDS_S1'),
                                     output_features=output_ASC_features,
                                     map_functions=S1_stats,
                                     percentiles=percentiles,
                                     axis=0)
    stats_DES = MapFeatureYearlyTask(input_feature=(FeatureType.DATA, 'BANDS_S1'),
                                     output_features=output_DES_features,
                                     map_functions=S1_stats,
                                     percentiles=percentiles,
                                     axis=0)

    # save S1 EOPatch
    save_S1_ASC = SaveTask(eopatches_S1_ASC_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)
    save_S1_DES = SaveTask(eopatches_S1_DES_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    # move S1 statistics to S2 EOPatch
    move_S1_ASC_to_S2 = MoveFeatureTask(FeatureType.DATA_TIMELESS)
    move_S1_DES_to_S2 = MoveFeatureTask(FeatureType.DATA_TIMELESS)

    # save updated EOPatch with S2 and S1 data
    save_S1_S2 = SaveTask(path=eopatches_folder, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

    """
    =================================================================================
    Creating EOWorkflow, then using EOExecutor to loop through areas (all EONodes)
    =================================================================================
    """

    # New workflow with nodes, EONodes
    node1 = EONode(create_eopatch, [], 'Create S2 EOPatch')
    node2 = EONode(add_S2_data, [node1], 'Add S2 data')
    node3 = EONode(S2_custom_CLM, [node2], 'Add your own cloud mask')
    node4 = EONode(valid_mask, [node3], 'Add "valid data" mask')
    node5 = EONode(save_S2, [node4], 'Save S2 EOPatch')
    node6 = EONode(add_S1_ASC_data, [], 'Add S1_ASC data')
    node7 = EONode(stats_ASC, [node6], 'Stats S1_ASC')
    node8 = EONode(save_S1_ASC, [node7], 'Save S1_ASC EOPatch')
    node9 = EONode(add_S1_DES_data, [], 'Add S1_DES data')
    node10 = EONode(stats_DES, [node9], 'Stats S1_DES')
    node11 = EONode(save_S1_DES, [node10], 'Save S1_DES EOPatch')
    node12 = EONode(move_S1_ASC_to_S2, [node7, node4], 'Move S1_ASC stats to S2 EOPatch')
    node13 = EONode(move_S1_DES_to_S2, [node9, node12], 'Move S1_DES stats to S2 EOPatch')
    node14 = EONode(save_S1_S2, [node13], 'Save EOPatch S1 and S2')

    workflow = EOWorkflow([node1, node2, node3, node4, node5,  # S2 EOPatch
                           node6, node7, node8,                # S1_ASC EOPatch
                           node9, node10,                      # S1_DES EOPatch
                           node11, node12, node13, node14])    # Save EOPatch S1 and S2


    execution_args = []
    for bbox_id, bbox_coords in list_of_bboxes:
        execution_args.append({
            node1: {'bbox': BBox(bbox_coords, crs=crs), 'timestamps': list_of_dates},
            node2: {'bbox': BBox(bbox_coords, crs=crs)},
            node6: {'bbox': BBox(bbox_coords, crs=crs), 'time_interval': time_interval},
            node9: {'bbox': BBox(bbox_coords, crs=crs), 'time_interval': time_interval},
            node5: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)}, #, 'bbox': BBox(bbox_coords, crs=crs)},
            node8: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)}, #, 'bbox': BBox(bbox_coords, crs=crs)},
            node11: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)}, #, 'bbox': BBox(bbox_coords, crs=crs)},
            node14: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)}, #, 'bbox': BBox(bbox_coords, crs=crs)},
        })

        '''
            create_eopatch: {'timestamp': list_of_dates},  # add dates to eopatch; request will be sent for these dates
            add_S2_data: {'bbox': BBox(bbox_coords, crs=crs)},
            add_S1_ASC_data: {'bbox': BBox(bbox_coords, crs=crs), 'time_interval': time_interval},
            add_S1_DES_data: {'bbox': BBox(bbox_coords, crs=crs), 'time_interval': time_interval},
            save_S2: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)},
            save_S1_ASC: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)},
            save_S1_DES: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)},
            save_S1_S2: {'eopatch_folder': 'eopatch_{}'.format(bbox_id)}'''

    ''' Solution from ARSET23 '''
    """

    # Create a splitter to obtain a list of bboxes with 5km sides
    bbox_splitter = UtmZoneSplitter([country_shape], country.crs, 5000)

    bbox_list = np.array(bbox_splitter.get_bbox_list())
    info_list = np.array(bbox_splitter.get_info_list())

    # Time interval for the SH request
    time_interval = ["2019-01-01", "2019-12-31"]

    # Define additional parameters of the workflow
    input_node = workflow_nodes[0]
    save_node = workflow_nodes[-1]
    execution_args = []
    for idx, bbox in list_of_bboxes:
        execution_args.append(
            {
                input_node: {"bbox": bbox, "time_interval": time_interval},
                save_node: {"eopatch_folder": f"eopatch_{idx}"},
            }
        )
    ''' end of SOLUTION'''
    """

    # TODO: SHDeprecationWarning: The string representation of `BBox` will change to match its `repr` representation value = format % v
    print('Now creating {} EOPatches...\n'.format(len(list_of_bboxes)))
    executor = EOExecutor(workflow, execution_args, save_logs=True, logs_folder=reports_folder)
    executor.run(workers=10, multiprocess=False)
    executor.make_report()

    print(f"Report was saved to location: {executor.get_report_path()}")

    time_end = datetime.datetime.now()
    time_duration = time_end - time_start
    print('\nExecution finished in: ', time_duration)
