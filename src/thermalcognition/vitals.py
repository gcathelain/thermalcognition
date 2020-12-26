# -*- coding: utf-8 -*-
"""

This module contains functions for extracting respiratory and heart rate from thermal videos.

:copyright: (c) 2020 EPHE
:license: MIT License, see LICENSE for details
:author: Guillaume Cathelain
:organization: EPHE
:contact: guillaume.cathelain@gmail.com
:date: 22/12/2020
:version: 0.0
"""
import numpy as np
import scipy.interpolate
from tqdm import tqdm


def roi_fourier_1dspectrum(video, mode="mean"):
    """
    Get 1d Fourier spectrum of a ROI video, to find fundamentals of heart rate and respiratory rate.

    :param video:   (n, h, w) ndarray
                        Containing the temperatures of the region of interest
    :param mode:        "mean", "median", "min", or "max"
                        Determines the selected statistics of each ROI frame
    :return:            1d array
                        Magnitude of the unilateral spectrum of the ROI 1d statistics
    """
    temperature_array = get_1dstatistics(video, mode=mode)
    temperature_fft = np.fft.fft(temperature_array - np.mean(temperature_array))
    temperature_fft = temperature_fft[:len(video) // 2]
    temperature_fft = np.abs(temperature_fft)
    return temperature_fft


def get_1dstatistics(video, mode="max"):
    """
    Get 1d statistics of a ROI video, frame by frame

    :param video:   (n, h, w) ndarray
                        Contains the temperatures of the region of interest
    :param mode:        "mean", "median", "min", or "max"
                        Determines the selected statistics of each ROI frame
    :return:            1d array
                        Statistics of the ROI video
    """
    if mode == "mean":
        temperature_array = np.mean(video.reshape(len(video), -1), axis=-1)
    elif mode == "median":
        temperature_array = np.median(video.reshape(len(video), -1), axis=-1)
    elif mode == "min":
        temperature_array = np.min(video.reshape(len(video), -1), axis=-1)
    elif mode == "max":
        temperature_array = np.max(video.reshape(len(video), -1), axis=-1)
    else:
        raise NotImplementedError()
    return temperature_array


def extrema_trajectory(video, mode="max"):
    """
    Check if extrema temperature of the video have a stable position in time.

    :param video:       (n, h, w) array
                        Contains the temperatures of the region of interest
    :param mode:        "min", or "max"
                        Determines the selected statistics of each ROI frame
    :return:            (n, 3) array
                        Contains row indices, column indices, and temperature values of extrema in time
    """
    flat_video = video.reshape(len(video), -1)
    if mode == "max":
        flat_indices = np.argmax(flat_video, axis=-1)
        values = np.max(flat_video, axis=-1)
    elif mode == "min":
        flat_indices = np.argmin(flat_video, axis=-1)
        values = np.min(flat_video, axis=-1)
    else:
        raise NotImplementedError()
    rows, cols = np.unravel_index(flat_indices, shape=video.shape[1:])

    return np.stack(rows, cols, values)


def get_line_temp(index, videodata, landmarks_df, landmark_i=36, landmark_j=31, line_resolution=100):
    """
    Get the temperature of a line between two landmarks.

    :param index:               int
                                Index of the frame
    :param videodata:           (n, h, w) array
                                Contains the temperatures of the region of interest
    :param landmarks_df:        pd.DataFrame
                                Contains landmarks position at every frame
    :param landmark_i:          int
                                Index of a landmark, default to right side of nose
    :param landmark_j:          int
                                Index of a landmark, default to left side of right eye
    :param line_resolution:     int
                                Number of interpolated points of the line from landmark_i to landmark_j
    :return:                    (n, line_resolution) ndarray
                                Temperature line.
    """
    frame_index = int(landmarks_df.loc[index, "frame_index"])
    frame = videodata[frame_index]
    pts = landmarks_df.iloc[index, 1:].to_numpy().reshape((-1, 2))
    line_x = np.linspace(pts[landmark_i, 0], pts[landmark_j, 0], line_resolution)
    line_y = np.linspace(pts[landmark_i, 1], pts[landmark_j, 1], line_resolution)
    xi, yi = pts[landmark_i, :].astype("int")
    xj, yj = pts[landmark_j, :].astype("int")
    x_low, x_up = sorted([xi, xj])
    y_low, y_up = sorted([yi, yj])
    x_up += 1
    y_up += 1
    frame_temperature = frame[y_low:y_up, x_low:x_up]
    if np.sum(np.array(frame_temperature.shape) < 2) > 0:
        temp_line = np.nan + np.zeros(line_resolution)
    else:
        try:
            interp_func = scipy.interpolate.interp2d(np.arange(x_low, x_up), np.arange(y_low, y_up), frame_temperature)
            temp_matrix = interp_func(line_x, line_y)
            temp_line = np.diag(temp_matrix)
        except ValueError:
            temp_line = np.zeros((line_resolution,)) + np.nan
    return temp_line


def get_matrix_temp(videodata, landmarks_df, landmark_i=36, landmark_j=31, line_resolution=100):
    """
    Concatenation of several temperature lines

    :param videodata:           (n, h, w) array
                                Contains the temperatures of the region of interest
    :param landmarks_df:        pd.DataFrame
                                Contains landmarks position at every frame
    :param landmark_i:          int
                                Index of a landmark, default to right side of nose
    :param landmark_j:          int
                                Index of a landmark, default to left side of right eye
    :param line_resolution:     int
                                Number of interpolated points of the line from landmark_i to landmark_j
    :return:                    (n, line_resolution) ndarray
                                Matrix of temperature lines along time.
    """
    temp_matrix = np.nan + np.zeros((len(landmarks_df["frame_index"]), line_resolution), dtype=np.float16)
    for n, index in tqdm(enumerate(landmarks_df["frame_index"].astype("int"))):
        temp_matrix[n] = get_line_temp(
            index, videodata, landmarks_df,
            landmark_i=landmark_i, landmark_j=landmark_j, line_resolution=line_resolution)
    return temp_matrix


def filter_physio(temp_matrix, fps=100):
    """
    Filters physiological activities in the temperature matrix

    :param temp_matrix: (n, line_resolution)
    :param fps:         float
                        Sampling rate
    :return:            (n,) ndarray
                        Maximum of the temperature segments, filtered in the 0.1 to 5 Hz frequency band.
    """
    raw_max_temperature = np.nanmax(temp_matrix, axis=-1)
    mask = ~np.isnan(raw_max_temperature)
    raw_max_temperature = scipy.interpolate.interp1d(np.arange(len(raw_max_temperature))[mask],
                                                     raw_max_temperature[mask])(np.arange(len(raw_max_temperature)))
    max_temperature = scipy.signal.medfilt(raw_max_temperature, 3)
    b, a = scipy.signal.butter(3, np.array([0.1, 5]) / (fps / 2), btype="bandpass")
    max_temperature = scipy.signal.filtfilt(b, a, max_temperature)
    return max_temperature
