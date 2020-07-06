

import h5py


dataset_id = "02f50f3f-4ec9-4192-a1f4-06ea93d50922"
cpdata = h5py.File(f'raw_data/{dataset_id}/cpdata.h5', 'r')

cpdata.keys()
# ['image', 'meta', 'object']


### IMAGE ####
cpdata['image'].keys()
# <KeysViewHDF5 ['M', 'cFeatureName', 'rImageNumber', 'rMetadata']>

cpdata['image']['M']
# <HDF5 dataset "M": shape (683, 2779), type "<f8">


cpdata['image']['M'][200,240]
# 0.000246685596513

cpdata['image']['cFeatureName']
# <HDF5 dataset "cFeatureName": shape (683,), type "|S51">

cpdata['image']['rImageNumber']
# <HDF5 dataset "rImageNumber": shape (2779,), type "<i4">

cpdata['image']['rMetadata']
# <HDF5 dataset "rMetadata": shape (2779,), type "|V3107">
# each entry is a bunch of paths and other knicknacks


### META ###
cpdata['meta'].keys()
# <KeysViewHDF5 ['experiment', 'imaging_run', 'processing_run']>

cpdata['meta']['experiment'].keys()
# <KeysViewHDF5 ['assay_plate_barcode', 'cell_sample_name', 'cell_staining_protocol_name', 'cell_staining_protocol_version', 'control_id', 'cpd_plate_nickname', 'cpd_platemap_name', 'dilution_factor', 'platemap', 'project', 'researcher_username', 'treatment_date', 'treatment_duration_hhmmss']>

cpdata['meta']['imaging_run'].keys()
# <KeysViewHDF5 ['channels', 'image_location', 'imaging_run_name', 'imaging_run_pass', 'imaging_run_timestamp', 'microscope_name', 'plate_read_id', 'plate_read_timestamp', 'plate_read_valid']>

cpdata['meta']['processing_run'].keys()
# <KeysViewHDF5 ['batch_info', 'processor_name', 'processor_version', 'protocol', 'protocol_analysis_file', 'protocol_illum_file', 'protocol_name', 'protocol_version']>


### OBJECT ###
cpdata['object'].keys()
# <KeysViewHDF5 ['M', 'cFeatureName', 'rImageNumber', 'rObjectNumber']>

cpdata['object']['M']
# <HDF5 dataset "M": shape (2198, 291249), type "<f8">

cpdata['object']['cFeatureName']
# <HDF5 dataset "cFeatureName": shape (2198,), type "|S59">

cpdata['object']['rImageNumber']
# <HDF5 dataset "rImageNumber": shape (291249,), type "<i4">

cpdata['object']['rObjectNumber']
# <HDF5 dataset "rObjectNumber": shape (291249,), type "<i4">
