import sys
import warnings
import re
import pytz
import math
import pprint as pp
import numpy as np
from pathlib import Path
from datetime import datetime

from pypl2lib import *

from pynwb import NWBFile, NWBHDF5IO, ProcessingModule, TimeSeries
from pynwb.ecephys import ElectricalSeries, LFP, EventDetection, FilteredEphys


class InconsistentInputException(Exception):
    pass


class UnsupportedPL2InputException(Exception):
    pass


def print_error(pypl2_file_reader_instance):
    """ Print a Plexon PL2 API error """
    error_message = (c_char * 256)()
    pypl2_file_reader_instance.pl2_get_last_error(error_message, 256)
    print(error_message.value)

def ctypes_struct_to_dict(struct):
    """
    Convert a ctypes Structure into a dictionary.
    Arrays are converted to lists using numpy.
    Strings are decoded in ASCII.
    Based on https://stackoverflow.com/a/3789491

    Args:
        struct - ctypes Structure with _fields_ property

    Returns:
        Dictionary corresponding to the key-value pairs in struct. This may have
        nested dictionaries if the input Structure had nested Structures.
    """
    result = {}
    for field, _ in struct._fields_:
         value = getattr(struct, field)
         if hasattr(value, '_length_') and hasattr(value, '_type_'):
             # Probably an array
             value = np.ctypeslib.as_array(value).tolist()
         elif type(value) is bytes:
             # Probably a string
             value = value.decode('ascii')
         elif hasattr(value, '_fields_'):
             # Probably another struct
             value = ctypes_struct_to_dict(value)
         result[field] = value
    return result

def print_pl2_info(pl2_object):
    """ Pretty print a PL2 object such as file info or channel info """
    pp.pprint(ctypes_struct_to_dict(pl2_object))

def tm_to_datetime(tm):
    """
    Convert a ctypes Structure representing a date-time from the Plexon PL2
    Python API into a datetime.

    Args:
        tm - ctypes Structure representing Plexon PL2 date-time

    Returns:
        datetime with year, month, day, hour, minute, second specified
    """
    # Plexon stores years relative to 1900, and month index starts at 0
    return datetime(tm.tm_year + 1900,
                    tm.tm_mon + 1,
                    tm.tm_mday,
                    tm.tm_hour,
                    tm.tm_min,
                    tm.tm_sec)

def get_channel_id(achan_name):
    """ Extract the channel ID from a PL2 analog/spike channel name """
    # note: we cannot use achannel_info.m_Channel because it is not
    # unique. but the number after 'FP' or 'SPK' in the channel name
    # *should* be unique and an integer
    match = re.search('\d', achan_name)
    if match:
        return int(achan_name[match.start():])
    else:
        warnings.warn('Channel name missing number: ' + achan_name)

def get_channel_ids_by_metadata_list(channel_ids, electrode_group_metadata):
    """
    Returns a list of channel ids out of those given that are present in each
    electrode group

    Args:
        channel_ids - list of scalars representing physical electrode channels
        electrode_group_metadata - list of dictionaries containing metadata for
            each electrode group. requires 'channel_ids' field indicating which
            channel ids are part of the corresponding electrode group

    Returns:
        list of which channel ids are found in which electrode group
    """
    electrode_group_to_channel_ids = []
    for egm in electrode_group_metadata:
        electrode_group_to_channel_ids.append([ch for ch in channel_ids if ch in egm['channel_ids']])
    return electrode_group_to_channel_ids

def find_electrode_by_id(electrodes, id):
    """
    Returns the index of the electrode with the given id, or None if not found

    Args:
        electrodes - ElectrodeTable(DynamicTable) instance corresponding
            to electrodes added to a NWBFile instance
        id - scalar to search for in the IDs of ElectrodeTable

    Returns:
        index of the corresponding electrode in the ElectrodeTable, or None if
        not found
    """
    if not electrodes:
        return None
    try:
        # technically returns index of the first match, but there should not be
        # more than one match, i.e. ids should be maintained as unique, though
        # this is not enforced in PyNWB
        return electrodes.id.data.index(id)
    except ValueError:
        return None

def add_electrodes(nwbfile, channel_ids, electrode_group):
    """
    Adds electrodes with the given ElectrodeGroup to the ElectrodeTable of the
    NWBFile instance. Does nothing if the channel id already exists in the
    ElectrodeTable. Electrode location is automatically set as the location of
    the ElectrodeGroup

    TODO: x, y, z, impedance, filtering are currently nan or 'none'

    Args:
        nwbfile - the NWBFile instance to add the electrode region to
        channel_ids - list of scalars representing physical electrode channels.
            these will become the unique ids for the electrodes
        electrode_group - ElectrodeGroup instance representing a set of
            electrodes that contain all of the electrodes being added. Usually
            corresponds to a physical electrode array or shank. An electrode
            can belong to only one electrode group.
    """

    for id in channel_ids:
        if find_electrode_by_id(nwbfile.electrodes, id) is None:
            print(f'Adding electrode corresponding to channel id {id}')
            nwbfile.add_electrode(x=math.nan,
                                  y=math.nan,
                                  z=math.nan,
                                  imp=math.nan,
                                  location=electrode_group.location,
                                  filtering='none',
                                  group=electrode_group,
                                  id=id)

def pl2_create_electrode_region(nwbfile, channel_ids, description):
    """
    Creates an electrode table region for the selected electrode channels of the
    NWBFile instance. Electrodes with those channel ids must have been first
    added to the ElectrodeTable of the NWBFile. Basically creates a vector of
    indices into the ElectrodeTable, useful when adding an ElectricalSeries
    containing a 2-D array of values for an arbitrary set of electrodes over
    time.

    Args:
        nwbfile - the NWBFile instance to add the electrode region to
        channel_ids - iterable of scalars representing physical electrode
            channels
        description - string describing the electrode table region

    Returns:
        DynamicTableRegion instance with indices into the ElectrodeTable of the
        NWBFile instance
    """

    electrode_inds = [find_electrode_by_id(nwbfile.electrodes, ch) for ch in channel_ids]
    if electrode_inds:
        print(f'Creating electrode region with description "{description}" for ElectrodeTable indices [' +
              ', '.join(str(x) for x in electrode_inds) + '] mapped to Plexon channel ids [' +
              ', '.join(str(x) for x in channel_ids) + ']')
    else:
        warnings.warn(f'Creating electrode region with description "{description}" with no electrodes')

    return nwbfile.create_electrode_table_region(electrode_inds, description)

def pl2_extract_timeseries_data(file_reader, file_handle, timestamp_frequency,
                                pl2_inds, electrodes=None):
    """
    Read analog time series data from a Plexon PL2 file and returns a dictionary
    containing the values as well as metadata for use in instantiating a
    TimeSeries (or ElectricalSeries)

    Args:
        file_reader - PyPL2FileReader instance
        file_handle - handle of the Plexon PL2 file being read
        timestamp_frequency - frequency in Hz of timestamps used by PL2 file
        pl2_inds - zero-based indices of the PL2 analog channels
        electrodes - DynamicTableRegion TODO

    Returns:
        a dictionary of keyword arguments for instantiating a TimeSeries
    """
    num_values_check = None
    num_fragments_returned_check = None
    num_data_points_returned_check = None
    fragment_ts_inds_check = None
    fragment_counts_check = None

    all_values = list()
    all_orig_names = list()

    for pl2i, i in zip(pl2_inds, range(len(pl2_inds))):
        # get the channel header information
        achannel_info = PL2AnalogChannelInfo()
        file_reader.pl2_get_analog_channel_info(file_handle, pl2i, achannel_info)

        achan_name = achannel_info.m_Name.decode('ascii')
        channel_id = get_channel_id(achan_name)

        if electrodes and channel_id != electrodes.table.id[electrodes.data[i]]:
            raise InconsistentInputException()

        # handling of encoding data not stored in volts is not yet implemented
        # ElectricalSeries() strictly uses Volts
        units = achannel_info.m_Units.decode('ascii')
        if units != 'Volts':
            raise UnsupportedPL2InputException()

        all_orig_names.append(achan_name)

        # get the data for this channel and basic info about the data
        print(f'Reading PL2 data for analog channel {pl2i} named "{achan_name}"...')
        num_fragments_returned = c_ulonglong(0)
        num_data_points_returned = c_ulonglong(0)
        fragment_ts_inds = (c_longlong * achannel_info.m_MaximumNumberOfFragments)()
        fragment_counts = (c_ulonglong * achannel_info.m_MaximumNumberOfFragments)()
        values = (c_short * achannel_info.m_NumberOfValues)()
        res = file_reader.pl2_get_analog_channel_data(file_handle,
                                                      pl2i,
                                                      num_fragments_returned,
                                                      num_data_points_returned,
                                                      fragment_ts_inds,
                                                      fragment_counts,
                                                      values)

        if (res == 0):
            print_error(file_reader)

        # convert from ctypes to Python/Numpy types
        num_fragments_returned = num_fragments_returned.value
        num_data_points_returned = num_data_points_returned.value
        fragment_ts_inds = np.ctypeslib.as_array(fragment_ts_inds)
        fragment_counts = np.ctypeslib.as_array(fragment_counts)

        # check that all properties except for the values are the same across
        # analog channels. otherwise, creating an ndarray with all channels will
        # be problematic, e.g. it may have inconsistent dimensions
        if num_values_check is None:
            num_values_check = achannel_info.m_NumberOfValues
            num_fragments_returned_check = num_fragments_returned
            num_data_points_returned_check = num_data_points_returned
            fragment_ts_inds_check = fragment_ts_inds
            fragment_counts_check = fragment_counts
        else:
            if num_values_check != achannel_info.m_NumberOfValues:
                raise InconsistentInputException()
            if num_fragments_returned_check != num_fragments_returned:
                raise InconsistentInputException()
            if num_data_points_returned_check != num_data_points_returned:
                raise InconsistentInputException()
            if (fragment_ts_inds_check != fragment_ts_inds).any():
                raise InconsistentInputException()
            if (fragment_counts_check != fragment_counts).any():
                raise InconsistentInputException()

        # why would input m_MaximumNumberOfFragments == 18 and num_fragments_returned == 1?
        fragment_ts = fragment_ts_inds[:num_fragments_returned] / timestamp_frequency
        if num_fragments_returned > 1:
            fragment_start_inds = [0] + np.cumsum(fragment_counts[:-1])[:num_fragments_returned].tolist()
        else:
            fragment_start_inds = [0]

        if np.sum(fragment_counts) != num_data_points_returned:
            raise InconsistentInputException()

        # TODO watch memory -- using a pointer may be more efficient
        all_values.append(np.ctypeslib.as_array(values))

    if all_values:
        # TimeSeries requires either timestamps or starting_time and rate
        return {'data': np.array(all_values).T,  # check performance
                'resolution': math.nan,
                'conversion': achannel_info.m_CoeffToConvertToUnits,
                'starting_time': fragment_ts[0]/achannel_info.m_SamplesPerSecond,
                'rate': achannel_info.m_SamplesPerSecond,
                'comments': ', '.join(all_orig_names)}

def pl2_create_electrode_timeseries(nwbfile, file_reader, file_handle,
                                    timestamp_frequency, channels, name,
                                    electrodes_desc):
    """
    Creates a single ElectricalSeries from analog channel data in the Plexon PL2
    file corresponding to electrode recordings. The PL2 data can be from
    multiple channels.

    TODO: add metadata about fragment_start_inds, fragment_ts to es

    Args:
        nwbfile - the NWBFile instance to add the electrode region to
        file_reader - PyPL2FileReader instance
        file_handle - handle of the Plexon PL2 file being read
        timestamp_frequency - frequency in Hz of timestamps used by PL2 file
        channels - list of dictionaries with fields:
            'pl2_ind' - zero-based index of the PL2 analog channel
            'channel_id' - scalar representing the physical electrode channel
        name - name to assign the ElectricalSeries
        electrodes_desc - description of the group of electrodes

    Returns:
        ElectricalSeries instance with the corresponding data and metadata
    """

    pl2_inds = [ch['pl2_ind'] for ch in channels]
    channel_ids = [ch['channel_id'] for ch in channels]

    # create a DynamicTableRegion selecting electrodes from the ElectrodeTable
    electrodes = pl2_create_electrode_region(nwbfile, channel_ids, electrodes_desc)

    adata = pl2_extract_timeseries_data(file_reader, file_handle,
                                        timestamp_frequency, pl2_inds,
                                        electrodes=electrodes)
    if adata:
        print(f'Creating electrical series named "{name}" for PL2 analog channels [' +
              ", ".join(str(x) for x in pl2_inds) + '], mapped to ElectrodeTable indices [' +
              ", ".join(str(x) for x in electrodes.data) + ']')
        return ElectricalSeries(name=name,
                                description=electrodes_desc,  # place extra desc here
                                electrodes=electrodes,
                                **adata)

def pl2_create_timeseries(nwbfile, file_reader, file_handle,
                          timestamp_frequency, pl2_inds, name, description):
    """
    Creates a single TimeSeries from analog channel data in the Plexon PL2
    file. The PL2 data can be from multiple channels.

    TODO: add metadata about fragment_start_inds, fragment_ts to es

    Args:
        nwbfile - the NWBFile instance to add the electrode region to
        file_reader - PyPL2FileReader instance
        file_handle - handle of the Plexon PL2 file being read
        timestamp_frequency - frequency in Hz of timestamps used by PL2 file
        pl2_inds - zero-based indices of the PL2 analog channels
        name - name to assign the TimeSeries

    Returns:
        TimeSeries instance with the corresponding data and metadata
    """

    adata = pl2_extract_timeseries_data(file_reader, file_handle,
                                        timestamp_frequency, pl2_inds)
    if adata:
        # need to specify either timestamps or starting_time and rate
        print(f'Creating time series named "{name}" for PL2 analog channels [' +
              ", ".join(str(x) for x in pl2_inds) + ']')
        return TimeSeries(name=name,
                          unit='volt',
                          description=description,  # place extra desc here
                          **adata)

def pl2_add_units(nwbfile, file_reader, file_handle, pl2_ind):
    """
    Adds spike times, waveforms, and metadata for all units on the given Plexon
    spike channel to the NWBFile instance

    Args:
        nwbfile - the NWBFile instance to add the electrode region to
        file_reader - PyPL2FileReader instance
        file_handle - handle of the Plexon PL2 file being read
        pl2_ind - zero-based index of the PL2 spike channel to read from
    """
    # get spike channel header info
    schannel_info = PL2SpikeChannelInfo()
    res = file_reader.pl2_get_spike_channel_info(file_handle, pl2_ind, schannel_info)

    if (res == 0):
        print_error(file_reader)

    if (schannel_info.m_ChannelEnabled and schannel_info.m_ChannelRecordingEnabled
        and schannel_info.m_NumberOfSpikes > 0):
        schan_name = schannel_info.m_Name.decode('ascii')
        channel_id = get_channel_id(schan_name)
        electrode_table_ind = find_electrode_by_id(nwbfile.electrodes, channel_id)

        # get spike channel data from PL2 file
        num_spikes_returned = c_ulonglong()
        spike_timestamps = (c_ulonglong * schannel_info.m_NumberOfSpikes)()
        unit_ids = (c_ushort * schannel_info.m_NumberOfSpikes)()
        values = (c_short * (schannel_info.m_NumberOfSpikes * schannel_info.m_SamplesPerSpike))()
        res = file_reader.pl2_get_spike_channel_data(file_handle,
                                           pl2_ind,
                                           num_spikes_returned,
                                           spike_timestamps,
                                           unit_ids,
                                           values)

        if (res == 0):
            print_error(file_reader)

        # Convert all A/D samples in 'values' to numpy array and then to volts
        # TODO watch memory usage here
        values = np.ctypeslib.as_array(values) * schannel_info.m_CoeffToConvertToUnits

        # 'values' is a 1-dimensional c_short() array of waveform samples, grouped by waveform
        # reshape the array into a 2D array of num_spikes x num_samples_per_spike
        values = values.reshape((schannel_info.m_NumberOfSpikes, schannel_info.m_SamplesPerSpike))

        spike_ts = np.ctypeslib.as_array(spike_timestamps) / schannel_info.m_SamplesPerSecond

        # 0 is unsorted, the rest go 1 to N
        unit_ids = np.ctypeslib.as_array(unit_ids)

        meas_units = schannel_info.m_Units.decode('ascii')
        if meas_units != 'Volts':
            raise UnsupportedPL2InputException()
        if len(spike_ts) != len(unit_ids):
            raise InconsistentInputException()
        if len(spike_ts) != values.shape[0]:
            raise InconsistentInputException()

        for j in range(schannel_info.m_NumberOfUnits):
            unit_ts = spike_ts[unit_ids == j]
            unit_wfs = values[unit_ids == j,]

            print(f'Adding unit {j} for channel {schannel_info.m_Channel} (' +
                  f'"{schannel_info.m_Name.decode("ascii")}"): {len(unit_ts)} waveforms')

            # add unit with auto-incrementing ID
            nwbfile.add_unit(id=None,
                             spike_times=unit_ts,
                             electrodes=[electrode_table_ind],  # one electrode per unit
                             pre_threshold_samples=schannel_info.m_PreThresholdSamples,
                             num_samples=schannel_info.m_SamplesPerSpike,
                             num_spikes=schannel_info.m_NumberOfSpikes,
                             Fs=schannel_info.m_SamplesPerSecond,
                             plx_sort_method=schannel_info.m_SortMethod,
                             plx_sort_range=(schannel_info.m_SortRangeStart, schannel_info.m_SortRangeEnd),
                             plx_sort_threshold=schannel_info.m_Threshold * schannel_info.m_CoeffToConvertToUnits,
                             is_unsorted=(j == 0),
                             channel_id=schannel_info.m_Channel,
                             waveforms=unit_wfs)


def main():
    # Get absolute path to file
    filename = Path(sys.argv[1]).resolve().as_posix()

    # TODO use with / as context manager - for handling open/closing files
    # TODO update the plexon API to use exceptions instead of this clunky
    # if result == 0 code
    file_reader = PyPL2FileReader()
    file_handle = file_reader.pl2_open_file(filename)

    if (file_handle == 0):
        print_error(file_reader)

    # create the PL2FileInfo instance containing basic header information
    file_info = PL2FileInfo()
    res = file_reader.pl2_get_file_info(file_handle, file_info)

    if (res == 0):
        print_error(file_reader)

    # USER NEEDS TO INPUT:

    # create the NWBFile instance
    session_description = 'Pulvinar recording from McCartney'
    id = 'M20170127'
    session_start_time = tm_to_datetime(file_info.m_CreatorDateTime)
    timezone = pytz.timezone("America/New_York")
    experimenter = 'Ryan Ly'
    lab = 'Kastner Lab'
    institution = 'Princeton University'
    experiment_description = 'Neural correlates of visual attention across the pulvinar'
    session_id = id
    data_collection = file_info.m_CreatorComment.decode('ascii')

    session_start_time = timezone.localize(session_start_time)
    nwbfile = NWBFile(session_description=session_description,
                      identifier=id,
                      session_start_time=session_start_time,
                      experimenter=experimenter,
                      lab=lab,
                      institution=institution,
                      experiment_description=experiment_description,
                      session_id=session_id,
                      data_collection=data_collection)

    # TODO add in the reprocessor metadata from file_info

    # create a recording device instance
    plexon_device_name = file_info.m_CreatorSoftwareName.decode('ascii') + '_v' + \
                         file_info.m_CreatorSoftwareVersion.decode('ascii')
    plexon_device = nwbfile.create_device(name=plexon_device_name)

    eye_trac_device_name = 'ASL_Eye-trac_6_via_' + plexon_device_name
    eye_trac_device = nwbfile.create_device(name=eye_trac_device_name)

    lever_device_name = 'Manual_lever_via_' + plexon_device_name
    lever_device = nwbfile.create_device(name=lever_device_name)
    # TODO update Device to take in metadata information about the Device

    # create electrode groups representing single shanks or other kinds of data
    # create list of metadata for adding into electrode groups. importantly, sasdasdsads
    # these metadata
    electrode_group_metadata = []
    electrode_group_metadata.append({'name': '32ch-array',
                                    'description': '32-channel_array',
                                    'location': 'Pulvinar',
                                    'device': plexon_device,
                                    'channel_ids': range(1, 33)})
    electrode_group_metadata.append({'name': 'test_electrode_1',
                                    'description': 'test_electrode_1',
                                    'location': 'unknown',
                                    'device': plexon_device,
                                    'channel_ids': [97]})
    electrode_group_metadata.append({'name': 'test_electrode_2',
                                    'description': 'test_electrode_2',
                                    'location': 'unknown',
                                    'device': plexon_device,
                                    'channel_ids': [98]})
    electrode_group_metadata.append({'name': 'test_electrode_3',
                                    'description': 'test_electrode_3',
                                    'location': 'unknown',
                                    'device': plexon_device,
                                    'channel_ids': [99]})
    non_electrode_ts_metadata = []
    non_electrode_ts_metadata.append({'name': 'eyetracker_x_voltage',
                                     'description': 'eyetracker_x_voltage',
                                     'location': 'n/a',
                                     'device': eye_trac_device,
                                     'channel_ids': [126]})
    non_electrode_ts_metadata.append({'name': 'eyetracker_y_voltage',
                                     'description': 'eyetracker_y_voltage',
                                     'location': 'n/a',
                                     'device': eye_trac_device,
                                     'channel_ids': [127]})
    non_electrode_ts_metadata.append({'name': 'lever_voltage',
                                     'description': 'lever_voltage',
                                     'location': 'n/a',
                                     'device': lever_device,
                                     'channel_ids': [128]})

    # make an electrode group for every group of channel IDs specified
    electrode_groups = []
    map_electrode_group_to_channel_ids = []
    for egm in electrode_group_metadata:
        print(f'Creating electrode group named "{egm["name"]}"')
        eg = nwbfile.create_electrode_group(name=egm['name'],
                                            description=egm['description'],
                                            location=egm['location'],
                                            device=egm['device'])
        electrode_groups.append(eg)
        map_electrode_group_to_channel_ids.append(egm['channel_ids'])

    # group indices of analog channels in the Plexon file by type, then source
    wb_src_chans = {};
    fp_src_chans = {};
    spkc_src_chans = {};
    ai_src_chans = {};
    aif_src_chans = {};

    for pl2_ind in range(file_info.m_TotalNumberOfAnalogChannels):
        achannel_info = PL2AnalogChannelInfo()
        res = file_reader.pl2_get_analog_channel_info(file_handle, pl2_ind, achannel_info)
        if (res == 0):
            print_error(file_reader)
            break

        if (achannel_info.m_ChannelEnabled and
                achannel_info.m_ChannelRecordingEnabled and
                achannel_info.m_NumberOfValues > 0 and
                achannel_info.m_MaximumNumberOfFragments > 0):
            # store zero-based channel index and electrode channel id
            achan_name = achannel_info.m_Name.decode('ascii');
            if achan_name.startswith('WB'):
                src_chans = wb_src_chans
            elif achan_name.startswith('FP') and get_channel_id(achan_name) <= 125:
                src_chans = fp_src_chans
            elif achan_name.startswith('FP') and get_channel_id(achan_name) > 125:
                src_chans = ai_src_chans
            elif achan_name.startswith('SPKC'):
                src_chans = spkc_src_chans
            elif achan_name.startswith('AI'):
                src_chans = ai_src_chans
            elif achan_name.startswith('AIF'):
                src_chans = aif_src_chans
            else:
                warnings.warn('Unrecognized analog channel: ' + achan_name)
                break

            channel_id = get_channel_id(achan_name)

            # src_chans is a dict {Plexon source ID : list of dict ...
            # {'pl2_ind': analog channel ind, 'channel_id': electrode channel ID}}
            chans = {'pl2_ind': pl2_ind, 'channel_id': channel_id}
            if not achannel_info.m_Source in src_chans:
                src_chans[achannel_info.m_Source] = [chans]
            else:
                src_chans[achannel_info.m_Source].append(chans)

    # create electrodes and create a region of indices in the electrode table
    # corresponding to the electrodes used for this type of analog data (WB, FP,
    # SPKC, AI, AIF)
    if wb_src_chans:
        for src, chans in wb_src_chans.items():
            channel_ids_all = [ch['channel_id'] for ch in chans]
            channel_ids_by_group = get_channel_ids_by_metadata_list(
                    channel_ids_all, electrode_group_metadata)
            for channel_ids, eg in zip(channel_ids_by_group, electrode_groups):
                if channel_ids:
                    # find the mapped pl2 index again
                    group_chans = [c for c in chans if c['channel_id'] in channel_ids]
                    add_electrodes(nwbfile, channel_ids, eg)
                    wb_es = pl2_create_electrode_timeseries(nwbfile,
                                                            file_reader,
                                                            file_handle,
                                                            file_info.m_TimestampFrequency,
                                                            group_chans,
                                                            'Wideband_voltages_' + eg.name,
                                                            'Wideband electrodes, group ' + eg.name)
                    nwbfile.add_acquisition(wb_es)

    if fp_src_chans or spkc_src_chans:
        ecephys_module = nwbfile.create_processing_module(name='ecephys',
                                                          description='Processed extracellular electrophysiology data')

    if fp_src_chans:
        # LFP signal can come from multiple Plexon "sources"
        # for src, chans in fp_src_chans.items():
        #     channel_ids_all = [ch['channel_id'] for ch in chans]
        #     channel_ids_by_group = get_channel_ids_by_metadata_list(
        #             channel_ids_all, electrode_group_metadata)
        #
        #     # channel_ids_by_group is a list that parallels
        #     # electrode_groups. it contains a list of lists of channel_ids that
        #     # are used for this type and source of analog data which are part of
        #     # the corresponding electrode group
        #     # for example, if the ElectrodeGroup at index 2 contains electrodes
        #     # with the channel IDs 1-32, and there is analog channel data for
        #     # channel ID 4-6, then channel_ids_by_group[2] would have
        #     # [4, 5, 6].
        #     d = []
        #     for c in channel_ids_all:
        #         for i in range(len(map_channel_ids_to_electrode_groups)):
        #             if c in get_channel_ids_by_metadata_list[i]:
        #                 d[i].append(c)
        #
        #     for channel_ids, eg in zip(channel_ids_by_group, electrode_groups):
        for src, chans in fp_src_chans.items():
            channel_ids = [ch['channel_id'] for ch in chans]
            electrode_group_to_channel_ids = get_channel_ids_by_metadata_list(
                    channel_ids, electrode_group_metadata)
            for channel_ids, eg, egm in zip(electrode_group_to_channel_ids,
                                                  electrode_groups,
                                                  electrode_group_metadata):
                if channel_ids:
                    group_chans = [c for c in chans if c['channel_id'] in channel_ids]
                    add_electrodes(nwbfile, channel_ids, eg)
                    lfp_es = pl2_create_electrode_timeseries(nwbfile,
                                                             file_reader,
                                                             file_handle,
                                                             file_info.m_TimestampFrequency,
                                                             group_chans,
                                                             'LFP_voltages_' + eg.name,
                                                             ('LFP electrodes, group ' + eg.name +
                                                              '. Low-pass filtering at 200 Hz done online by Plexon data acquisition system.'))

                    print('Adding LFP processing module with electrical series for channel ids [' +
                          ', '.join(str(x) for x in channel_ids) + '] for electrode group ' +
                          eg.name)
                    # TODO add LFP filter properties, though these are not stored in the PL2
                    # file
                    lfp = LFP(lfp_es, name='LFP_' + egm['name'])
                    ecephys_module.add(lfp)

    spkc_es = None
    if spkc_src_chans:
        for src, chans in spkc_src_chans.items():
            channel_ids = [ch['channel_id'] for ch in chans]
            electrode_group_to_channel_ids = get_channel_ids_by_metadata_list(
                    channel_ids, electrode_group_metadata)
            for channel_ids, eg, egm in zip(electrode_group_to_channel_ids,
                                                  electrode_groups,
                                                  electrode_group_metadata):
                if channel_ids:
                    group_chans = [c for c in chans if c['channel_id'] in channel_ids]
                    add_electrodes(nwbfile, channel_ids, eg)
                    spkc_es = pl2_create_electrode_timeseries(nwbfile,
                                                              file_reader,
                                                              file_handle,
                                                              file_info.m_TimestampFrequency,
                                                              group_chans,
                                                              'High-pass_filtered_voltages_' + egm['name'],
                                                              ('High-pass filtered ("SPKC") electrodes, group ' + egm['name'] +
                                                               '. High-pass filtering at 300 Hz done online by Plexon data acquisition system.'))

                    print('Adding SPKC processing module with electrical series for channel ids [' +
                          ', '.join(str(x) for x in channel_ids) + '] for electrode group ' +
                          egm['name'])
                    # TODO add SPKC filter properties, though these are not stored in the PL2
                    # file
                    filt_ephys = FilteredEphys(spkc_es, name='SPKC_' + egm['name'])
                    ecephys_module.add(filt_ephys)

    if ai_src_chans:
        for src, chans in ai_src_chans.items():
            channel_ids_all = [ch['channel_id'] for ch in chans]
            channel_ids_by_group = get_channel_ids_by_metadata_list(
                    channel_ids_all, non_electrode_ts_metadata)
            for channel_ids, gm in zip(channel_ids_by_group, non_electrode_ts_metadata):
                if channel_ids:
                    # find the mapped pl2 index again
                    pl2_inds = [c['pl2_ind'] for c in chans if c['channel_id'] in channel_ids]
                    ai_es = pl2_create_timeseries(nwbfile,
                                                  file_reader,
                                                  file_handle,
                                                  file_info.m_TimestampFrequency,
                                                  pl2_inds,
                                                  ('Auxiliary_input_' + str(src) +
                                                   '_' + gm['name']),
                                                  ('Auxiliary input, source ' + str(src) +
                                                   ', ' + gm['name']))
                    nwbfile.add_acquisition(ai_es)

    if aif_src_chans:
        for src, chans in aif_src_chans.items():
            channel_ids_all = [ch['channel_id'] for ch in chans]
            channel_ids_by_group = get_channel_ids_by_metadata_list(
                    channel_ids_all, non_electrode_ts_metadata)
            for channel_ids, gm in zip(channel_ids_by_group, non_electrode_ts_metadata):
                if channel_ids:
                    # find the mapped pl2 index again
                    pl2_inds = [c['pl2_ind'] for c in chans if c['channel_id'] in channel_ids]
                    aif_es = pl2_create_timeseries(nwbfile,
                                                  file_reader,
                                                  file_handle,
                                                  file_info.m_TimestampFrequency,
                                                  pl2_inds,
                                                  ('Filtered_auxiliary_input_' + str(src) +
                                                   '_' + gm['name']),
                                                  ('Filtered auxiliary input, source ' + str(src) +
                                                   ', ' + gm['name']))
                    nwbfile.add_acquisition(aif_es)

    #### Spikes ####

    # add these columns to unit table
    nwbfile.add_unit_column('pre_threshold_samples', 'number of samples before threshold')
    nwbfile.add_unit_column('num_samples', 'number of samples for each spike waveform')
    nwbfile.add_unit_column('num_spikes', 'number of spikes')
    nwbfile.add_unit_column('Fs', 'sampling frequency')
    nwbfile.add_unit_column('plx_sort_method', 'sorting method encoded by Plexon')
    nwbfile.add_unit_column('plx_sort_range', 'range of sample indices used in Plexon sorting')
    nwbfile.add_unit_column('plx_sort_threshold', 'voltage threshold used by Plexon sorting')
    nwbfile.add_unit_column('is_unsorted', 'whether this unit is the set of unsorted waveforms')
    nwbfile.add_unit_column('channel_id', 'original recording channel ID')

    # since waveforms are not a 1:1 mapping per unit, use table indexing

    nwbfile.add_unit_column('waveforms', 'waveforms for each spike', index=True)

    # add a unit for each spike channel
    for i in range(file_info.m_TotalNumberOfSpikeChannels):
        pl2_add_units(nwbfile, file_reader, file_handle, i)

    # if spkc_series is not None:
    #     # Plexon does not save the indices of the spike times in the
    #     # high-pass filtered data. So work backwards from the spike times
    #     # first convert spike times to sample indices, accounting for imprecision
    #     spike_inds = spike_ts * schannel_info.m_SamplesPerSecond
    #
    #     # TODO this can be SUPER INEFFICIENT
    #     if not all([math.isclose(x, np.round(x)) for x in spike_inds]):
    #         raise InconsistentInputException()
    #
    #     spike_inds = np.round(spike_inds) # need to account for fragments TODO
    #
    #     ed_module = nwbfile.create_processing_module(name='Plexon online sorted units - all',
    #                                                  description='All units detected online')
    #     print('Adding Event Detection processing module for Electrical Series ' +
    #           f'named {spkc_series.name}')
    #     ed = EventDetection(detection_method="xx", # TODO allow user input
    #                         source_electricalseries=spkc_series,
    #                         source_idx=spike_inds,
    #                         times=spike_ts)
    #     ed_module.add(ed)

    # write NWB file to disk
    out_file = './output/nwb_test.nwb'
    with NWBHDF5IO(out_file, 'w') as io:
        print('Writing to file: ' + out_file)
        io.write(nwbfile)

    # Close the PL2 file
    file_reader.pl2_close_file(file_handle)

if __name__ == '__main__':
    main()
