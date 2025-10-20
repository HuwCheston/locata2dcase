# Implementation adapted from https://github.com/audiofhrozen/locata_wrapper/blob/master/locata_wrapper/utils/load_data.py
import csv
import glob
import logging
import os
import sys
from argparse import Namespace

import numpy as np
import pandas as pd
import soundfile


def wrapTo2Pi(_lambda):
    """Wrap angle in radians to [0 pi]"""
    positiveInput = _lambda > 0
    _lambda = np.mod(_lambda, 2 * np.pi)
    _idx = (_lambda == 0) * positiveInput
    _lambda[_idx] = 2 * np.pi
    return _lambda


def wrapToPi(_lambda):
    """Wrap angle in radians to [-pi pi]"""
    q = (_lambda < -np.pi) + (np.pi < _lambda)
    _lambda[q] = wrapTo2Pi(_lambda[q] + np.pi) - np.pi
    return _lambda


def cart2pol(cart):
    """cart2pol
    Cartesian to spherical transformation for LOCATA coordinate system
    Inputs:
        x:      Cartesian x-position [m]
        y:      Cartesian y-position [m]
        z:      Cartesian z-position [m]
    Outputs:
        az:     Azimuth [rad]
        el:     Elevation [rad]
        rad:      Radius [m]
    """

    pol = np.zeros(cart.shape)
    x = cart[:, 0]
    y = cart[:, 1]
    z = cart[:, 2]
    # radius
    pol[:, 2] = np.sqrt(np.abs(x) ** 2 + np.abs(y) ** 2 + np.abs(z) ** 2)
    # elev
    pol[:, 1] = np.rad2deg(np.arccos(z / pol[:, 2]) - np.pi/2)
    # azimuth
    pol[:, 0] = np.rad2deg(wrapToPi(np.arctan2(y, x) - (np.pi / 2)))
    return pol, cart


def load_wav(fnames, obj_type):
    obj = Namespace()
    obj.data = dict()
    for this_wav in fnames:
        # Load data:
        data, fs = soundfile.read(this_wav)

        # Array name:
        this_obj = os.path.basename(this_wav).replace('.wav', '')
        this_obj = this_obj.replace('{}_'.format(obj_type), '')

        # Load timestamps:
        _txt_table = this_wav.replace('{}.wav'.format(this_obj),
                                      'timestamps_{}.txt'.format(this_obj))
        txt_table = np.loadtxt(_txt_table, delimiter='\t', skiprows=1).T

        # Copy to namespace:
        obj.fs = fs
        obj.data[str(this_obj)] = data
        obj.time = txt_table
    return obj


def load_txt(fnames, obj_type):
    obj = Namespace()
    obj.data = dict()
    for this_txt in fnames:
        # Load data:
        txt_table = pd.read_csv(this_txt, sep='\t', header=0)
        _time = txt_table[['year', 'month', 'day',
                           'hour', 'minute', 'second']]
        _pos = txt_table[['x', 'y', 'z']].values.T
        _ref = txt_table[['ref_vec_x', 'ref_vec_y', 'ref_vec_z']].values.T
        _rot_1 = txt_table[['rotation_11', 'rotation_12', 'rotation_13']].values
        _rot_2 = txt_table[['rotation_21', 'rotation_22', 'rotation_23']].values
        _rot_3 = txt_table[['rotation_31', 'rotation_32', 'rotation_33']].values
        _rot = np.stack([_rot_1, _rot_2, _rot_3], axis=0)

        mics = list(set([x.split('_')[0] for x in txt_table if 'mic' in x]))
        if len(mics) > 0:
            for i in range(len(mics)):
                _lbl = ['mic{}_{}'.format(i + 1, x) for x in ['x', 'y', 'z']]
                _data = txt_table[_lbl].values.T
                if i == 0:
                    _mic = np.zeros((3, _data.shape[1], len(mics)))
                _mic[:, :, i] = _data
        else:
            _mic = None

        # Array name:
        this_obj = os.path.basename(this_txt).replace('.txt', '')
        this_obj = this_obj.replace('{}_'.format(obj_type), '')

        # Copy to namespace:
        obj.time = pd.to_datetime(_time)
        obj.data[str(this_obj)] = Namespace(
            position=_pos, ref_vec=_ref, rotation=_rot,
            mic=_mic)
    return obj


def load_txt_vad(fnames, obj_type):
    obj = Namespace()
    obj.data = dict()
    for this_txt in fnames:
        # Load data:
        txt_table = pd.read_csv(this_txt, sep='\t', header=0)
        _vad = txt_table[['VAD']].values.T

        mics = list(set([x.split('_')[0] for x in txt_table if 'mic' in x]))
        # Array name:
        this_obj = os.path.basename(this_txt).replace('.txt', '')
        this_obj = this_obj.replace('{}_'.format(obj_type), '')

        # Copy to namespace:
        obj.data[str(this_obj)] = Namespace(vad=_vad)
    return obj


def LoadData(this_array, args=None, log=logging, is_dev=True):
    """loads LOCATA csv metadata
    Inputs:
        dir_name:     Directory name containing LOCATA data (default: ../data/)
    Outputs:
        position_array:   Structure containing positional information of each of the arrays
        position_source:  Structure containing positional information of each source
        required_time:    Structure containing the timestamps at which participants must provide estimates
    """

    # Time vector:
    txt_array = pd.read_csv(os.path.join(this_array, 'required_time.txt'),
                            sep='\t')
    _time = pd.to_datetime(txt_array[['year', 'month', 'day',
                                      'hour', 'minute', 'second']])
    _valid = np.array(txt_array['valid_flag'].values, dtype=np.bool)
    required_time = Namespace(time=_time, valid_flag=_valid)

    # Audio files:
    wav_fnames = glob.glob(os.path.join(this_array, '*.wav'))

    audio_array_idx = [x for x in wav_fnames if 'audio_array' in x]
    if is_dev:
        audio_source_idx = [x for x in wav_fnames if 'audio_source' in x]
        if len(audio_array_idx) + len(audio_source_idx) == 0:
            log.error(f'Unexpected audio file in folder {this_array}')
            sys.exit(1)
    else:
        if len(audio_array_idx) == 0:
            log.error(f'Unexpected audio file in folder {this_array}')
            sys.exit(1)

    # Position source data:
    txt_fnames = glob.glob(os.path.join(this_array, '*.txt'))
    if is_dev:
        position_source_idx = [x for x in txt_fnames if 'position_source' in x]
        position_source = load_txt(position_source_idx, 'position_source')
    else:
        position_source = None

    # Voice activity data
    vad_source_idx = [x for x in txt_fnames if 'VAD_source' in x]
    vad_source = load_txt_vad(vad_source_idx, 'VAD_source')

    # Position array data:
    position_array_idx = [x for x in txt_fnames if 'position_array' in x]
    position_array = load_txt(position_array_idx, 'position_array')

    # Outputs:
    return position_array, position_source, required_time, vad_source


def GetTruth(this_array, position_array, position_source, vad_source, required_time, recording_id, is_dev=True):
    """GetLocataTruth
    creates Namespace containing OptiTrac ground truth data for and relative to the specified array
    Inputs:
        array_name:       String containing array name: 'eigenmike', 'dicit', 'benchmark2', 'dummy'
        position_array:   Structure containing array position data
        position_source:  Structure containing source position data
        required_time:    Vector of timestamps at which ground truth is required
        is_dev:           If 0, the evaluation database is considered and the
                          development database otherwise.
    Outputs:
        truth:            Namespace containing ground truth data
                          Positional information about the sound sources are
                          only returned for the development datbase
                          (is_dev = 1).
    """
    truth = Namespace()

    # Specified array
    truth.array = position_array.data[this_array]
    for field in truth.array.__dict__:
        _new_value = getattr(truth.array, field)[:, required_time.valid_flag]
        setattr(truth.array, field, _new_value)
    # Source
    if is_dev:
        frames = int(np.sum(required_time.valid_flag))
        truth.source = position_source.data
        truth.vad = vad_source.data
        # All sources for this recording
        for src_idx in truth.source:
            for field in truth.source[src_idx].__dict__:
                _new_value = getattr(truth.source[src_idx], field)
                if _new_value is not None:
                    setattr(truth.source[src_idx], field, _new_value[:, required_time.valid_flag])
            # Azimuth and elevation relative to microphone array
            h_p = truth.source[src_idx].position - truth.array.position
            # Apply rotation / translation of array to source
            pol_pos = np.squeeze(np.matmul(truth.array.rotation.transpose(1, 2, 0), h_p.T[:, :, None]))
            # Returned in azimuth, elevation & radius
            truth.source[src_idx].polar_pos, truth.source[src_idx].cart_pos = cart2pol(pol_pos)
            truth.frames = frames
    truth.recording_id = recording_id

    return truth


def ElapsedTime(time_array):
    n_steps = time_array.shape[0]
    elapsed_time = np.zeros([n_steps])
    for i in range(1, n_steps):
        elapsed_time[i] = (time_array[i] - time_array[i - 1]).total_seconds()
    return np.cumsum(elapsed_time)


def ProcessTaskMetadata(this_task, arrays, data_dir, is_dev):
    task_dir = os.path.join(data_dir, 'task{}'.format(this_task))
    # Read all recording IDs available for this task:
    recordings = sorted(glob.glob(os.path.join(task_dir, '*')))
    print("recordings", recordings)
    truth_list = []
    # Parse through all recordings within this task:
    for this_recording in recordings:
        recording_id = int(this_recording.split('recording')[1])
        # Read all recording IDs available for this task:
        array_names = glob.glob(os.path.join(this_recording, '*'))
        for array_dir in array_names:
            this_array = os.path.basename(array_dir)
            if this_array not in arrays:
                continue
            print('Processing task {}, recording {}, array {}.'.format(this_task, recording_id, this_array))
            # Load metadata from csv
            position_array, position_source, required_time, vad_source = LoadData(
                array_dir, None, None, is_dev)
            print('Processing Complete!')
            # Extract ground truth
            # position_array stores all optitrack measurements.
            # Extract valid measurements only (specified by required_time.valid_flag).
            truth = GetTruth(this_array, position_array, position_source, vad_source, required_time, recording_id, is_dev)
            truth_list.append(truth)
    return truth_list


def Locata2DecaseFormat(tasks, data_dir, coord_system="cartesian"):
    FS_POS = 120 # Position labeling done at 120Hz
    FS_AUD = 48000 # Sampling rate of original audio
    for task_id in tasks:
        truth_list = ProcessTaskMetadata(task_id, ["eigenmike"], data_dir, is_dev=True)
        for truth in truth_list:
            split = "dev" if "dev" in data_dir else "eval"
            out_filename = f'./output/{split}/task{task_id}_recording{truth.recording_id}.csv'
            # Silently skip existing files
            if os.path.isfile(out_filename):
                continue
            with open(out_filename, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                print("Processing {}".format(out_filename))
                for iframe in range(0, truth.frames, FS_POS//10): # sample every 100msec
                    for speak_num, speaker in enumerate(truth.source):
                        frame_num = iframe//12
                        total_len = len(list(truth.vad[speaker].vad[0]))
                        if total_len < (frame_num+1)*(FS_AUD//10):
                            upper_bnd = total_len
                        else:
                            upper_bnd = (frame_num+1)*(FS_AUD//10)
                        # Voting VAD to identify active speaker in a given frame
                        spk_vad = list(truth.vad[speaker].vad[0])[frame_num*(FS_AUD//10):upper_bnd]
                        spk_active = np.mean(spk_vad) > 0.5 # active if more than 50% is 1 within the frame
                        if not spk_active:
                            continue
                        csv_row = [frame_num, 0, speak_num]
                        if coord_system == "cartesian":
                            csv_row.extend(truth.source[speaker].cart_pos[iframe])
                        elif coord_system == "polar":
                            csv_row.extend(truth.source[speaker].polar_pos[iframe])
                        csv_writer.writerow(csv_row)


if __name__ == "__main__":
    Locata2DecaseFormat(["1", "2", "3", "4"], "./data/dev", coord_system="polar")
    Locata2DecaseFormat(["1", "2", "3", "4"], "./data/eval", coord_system="polar")
