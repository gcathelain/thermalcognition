# -*- coding: utf-8 -*-
"""

This module contains functions for calibration temperature in thermal videos with a temperature colorbar.

:copyright: (c) 2020 EPHE
:license: MIT License, see LICENSE for details
:author: Guillaume Cathelain
:organization: EPHE
:contact: guillaume.cathelain@gmail.com
:date: 22/12/2020
:version: 0.0
"""
import re
import pytesseract
import sklearn.linear_model
import scipy.interpolate
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2


def fit_temperature(rgb_image, temperature_image):
    """
    Fit linear regression from first frame of the video, when temperature array is directly available.

    :param rgb_image:           (h, w, 3) ndarray
                                (R,G,B) float features ranging from 0. to 1.
    :param temperature_image:   (h, w) ndarray
                                T float feature in °C degrees
    :return:                    scikit-learn linear regression model
    """
    colors_array = rgb_image.reshape((-1, 3))
    temperature_array = temperature_image.flatten()
    linreg_model = sklearn.linear_model.LinearRegression().fit(colors_array, temperature_array)
    return linreg_model


def predict_temperature(rgb_image, linreg_model):
    """
    Predict temperature of a frame using a linear regression model on the visible image.

    :param rgb_image:           (h, w, 3) ndarray
                                (R,G,B) float features ranging from 0. to 1.
    :param linreg_model:        scikit-learn linear regression model
                                Converts RGB to temperature
    :return:                    (h, w) ndarray
                                T float feature in °C degrees
    """
    colors_array = rgb_image.reshape((-1, 3))
    temperature_array = linreg_model.predict(colors_array)
    h, w, _ = rgb_image.shape
    return temperature_array.reshape((h, w))


def predict_temperature_video(rgb_videodata, linreg_model, batch_size=1000000):
    """
    Predict temperature of a video using a linear regression model on the visible image.

    :param rgb_videodata:           (n, h, w, 3) ndarray
                                (R,G,B) float features ranging from 0. to 1.
    :param linreg_model:        scikit-learn linear regression model
                                Converts RGB to temperature
    :param batch_size:          int
                                Number of pixels to process by iteration
    :return:                    (h, w) ndarray
                                T float feature in °C degrees
    """
    n, h, w, _ = rgb_videodata.shape
    start = np.arange(0, n * h * w // batch_size, dtype="int64") * batch_size
    temperature_video_flat = np.empty((n * h * w,))
    rgb_videodata_flat = rgb_videodata.reshape((-1, 3))
    for s in tqdm(start):
        colors_array = rgb_videodata_flat[s:s + batch_size]
        temperature_video_flat[s:s + batch_size] = linreg_model.predict(colors_array)
    return temperature_video_flat.reshape((n, h, w))


def fit_colorscale(image, y_low=17, y_up=-18, x_pos=340):
    """
    Get colorscale of a thermal frame, where part of the frame is a vertical colorbar for temperature.
    Default parameters are given for the Testo camera settings.

    :param image:   (h, w, 3) ndarray
                    (R,G,B) float features ranging from 0. to 1.
    :param y_low:   int
                    Vertical lower position of the colorbar
    :param y_up:    int
                    Vertical upper position of the colorbar
    :param x_pos:   int
                    Horizontal position of the colorbar
    :return:        scikit-learn linear regression model
    """
    colorbar = image[y_low:y_up, x_pos]
    colorbar = np.flip(colorbar, axis=0)
    temp_scale = np.linspace(0, 1, len(colorbar))
    linreg_model = sklearn.linear_model.LinearRegression().fit(colorbar, temp_scale)
    return linreg_model


def get_tempscale_df(videodata, text_size=(11, 35), up_text_pos=(1, 325), low_text_pos=(-16, 325)):
    """
    When colorbar changes in time, new temperature limits are recognized. Default values for Testo videos.

    :param videodata:       (n, h, w, 3) ndarray
                            RGB frames of a video including a colorbar
    :param text_size:       (h, w) tuple
                            Shape of the box where to look for digits
    :param up_text_pos:     (y, x) tuple
                            Location of the box related to the upper temperature limit of the colorbar
    :param low_text_pos:    (y, x) tuple
                            Location of the box related to the lower temperature limit of the colorbar
    :return:                pd.DataFrame
                            Containing frame indices of temperature changes and the colorbar limits values.
    """
    has_changed_indices = temperature_has_changed_vector(
        videodata, text_size=text_size, up_text_pos=up_text_pos, low_text_pos=low_text_pos)
    tempscale_df = pd.DataFrame(columns=["frame_index", "lower_temp", "upper_temp"])
    for index in tqdm(has_changed_indices):
        tempscale_df.loc[index, "frame_index"] = index
        tempscale_df.loc[index, "lower_temp"], tempscale_df.loc[index, "upper_temp"] = get_temperature_scale(
            videodata[index])
    tempscale_df = tempscale_df.astype({"frame_index": "int", "lower_temp": "float", "upper_temp": "float"})
    tempscale_df = tempscale_df.reset_index(drop=True)
    return tempscale_df


def temperature_has_changed_vector(videodata, text_size=(11, 35), up_text_pos=(1, 325), low_text_pos=(-16, 325)):
    """
    Return frame indices when the colorbar changes in time. Default values for Testo videos.

    :param videodata:       (n, h, w, 3) ndarray
                            RGB frames of a video including a colorbar
    :param text_size:       (h, w) tuple
                            Shape of the box where to look for digits
    :param up_text_pos:     (y, x) tuple
                            Location of the box related to the upper temperature limit of the colorbar
    :param low_text_pos:    (y, x) tuple
                            Location of the box related to the lower temperature limit of the colorbar
    :return:                ndarray
                            Indices of colorbar changes in time
    """
    h_text, w_text = text_size
    y_low_offset, x_offset = low_text_pos
    y_up_offset = up_text_pos[0]
    videodata_upper_temp_img = videodata[:, y_up_offset:y_up_offset + h_text, x_offset:x_offset + w_text, :]
    videodata_lower_temp_img = videodata[:, y_low_offset:y_low_offset + h_text, x_offset:x_offset + w_text, :]
    up_change_indices = frame_has_changed(videodata_upper_temp_img)
    low_change_indices = frame_has_changed(videodata_lower_temp_img)
    has_changed_indices = np.unique(np.sort(np.concatenate((up_change_indices, low_change_indices), axis=0)))
    return has_changed_indices


def frame_has_changed(videodata):
    """
    Return frame indices when the image changes in time.

    :param videodata:   (n, h, w, 3) ndarray
                        a video
    :return:            ndarray
                        Indices of image changes in time
    """
    has_changed_norm = np.sum(np.abs(np.diff(videodata, axis=0)), axis=(1, 2, 3))
    has_changed_indices = np.where(has_changed_norm != 0)[0]
    has_changed_indices = np.insert(1 + has_changed_indices, 0, 0)
    return has_changed_indices


def fill_tempscale_df(tempscale_df, videodata):
    """
    Fills a discrete temperature scale DataFrame with the previous temperature limits values.

    :param tempscale_df:    pd.DataFrame
                            Containing frame indices of temperature changes and the colorbar limits values.
    :param videodata:       (n, h, w, 3) ndarray
                            RGB frames of a video including a colorbar
    :return:                pd.DataFrame
                            Containing the lower and upper temperature in colorbar for every frame.
    """
    has_changed_indices = tempscale_df["frame_index"].to_numpy()
    frame_indices = np.arange(len(videodata))
    lower_temps = scipy.interpolate.interp1d(has_changed_indices, tempscale_df["lower_temp"].to_numpy(),
                                             kind="previous", fill_value="extrapolate")(frame_indices)
    upper_temps = scipy.interpolate.interp1d(has_changed_indices, tempscale_df["upper_temp"].to_numpy(),
                                             kind="previous", fill_value="extrapolate")(frame_indices)
    tempscale_df = pd.DataFrame(data=dict(frame_index=frame_indices, lower_temp=lower_temps, upper_temp=upper_temps))
    return tempscale_df


def rgb2temp_video(videodata, tempscale_df, thermal_image_shape=(240, 320), dtype=np.float16, batch_size=1000):
    """
    Converts visible video into temperature video

    :param videodata:           (n, h, w, 3) ndarray
                                RGB frames of a video including a colorbar
    :param tempscale_df:        pd.DataFrame
                                Containing frame indices of temperature changes and the colorbar limits values.
    :param thermal_image_shape: (h,w) tuple
                                Shape of the temperature image, excluding the colorbar, starting from upper left corner
    :param dtype:               numpy type
                                np.float16, np.float32, np.double, np.long depending of the coding of the temperature
                                values
    :param batch_size:          int
                                maximum number of frames used by iteration to avoid memory issues
    :return:                    (n, h, w) ndarray
                                Thermal frames of a video
    """
    h, w = thermal_image_shape
    linreg_model = fit_colorscale(videodata[0])
    videodata = videodata[:, :, :w]
    tempscale_df = fill_tempscale_df(tempscale_df, videodata)
    upper_temps = tempscale_df["upper_temp"].to_numpy()
    lower_temps = tempscale_df["lower_temp"].to_numpy()
    videodata_temperature = np.zeros(videodata.shape[:-1], dtype=dtype)
    n_batches = 1 + len(videodata) // batch_size
    for k in tqdm(range(n_batches)):
        batch_mask = range(k * batch_size, min((k + 1) * batch_size, len(videodata)))
        videodata_temperature[batch_mask] = linreg_model.predict(videodata[batch_mask].reshape((-1, 3))).reshape(
            (-1, h, w))
        videodata_temperature[batch_mask] *= (upper_temps[batch_mask] - lower_temps[batch_mask]).reshape((-1, 1, 1))
        videodata_temperature[batch_mask] += lower_temps[batch_mask].reshape((-1, 1, 1))
    return videodata_temperature


def get_temperature_scale(frame, interpolation_factor=4,
                          text_size=(11, 35), up_text_pos=(1, 325), low_text_pos=(-16, 325)):
    """
    Finds upper and lower limit of a Testo temperature frame.

    :param frame:                   (h,w, 3) ndarray
                                    RGB image with colorbar
    :param interpolation_factor:    int
                                    Interpolating digits makes it easier to recognize characters
    :param text_size:               (h, w) tuple
                                    Shape of the box where to look for digits
    :param up_text_pos:             (y, x) tuple
                                    Location of the box related to the upper temperature limit of the colorbar
    :param low_text_pos:            (y, x) tuple
                                    Location of the box related to the lower temperature limit of the colorbar
    :return:                        tuple
                                    (lower_temp, upper_temp) found in the frame
    """
    h_text, w_text = text_size
    y_low_offset, x_offset = low_text_pos
    y_up_offset = up_text_pos[0]
    upper_temp_img, lower_temp_img = [
        cv2.resize(frame[y_offset:y_offset + h_text, x_offset:x_offset + w_text, :],
                   (w_text * interpolation_factor, h_text * interpolation_factor), cv2.INTER_LINEAR)
        for y_offset in [y_up_offset, y_low_offset]
    ]
    upper_temp = get_temperature_float(upper_temp_img)
    lower_temp = get_temperature_float(lower_temp_img)
    return lower_temp, upper_temp


def get_temperature_float(img, config="--psm 10 --oem 3"):
    """
    Finds temperature float in image.

    :param img:     (h,w, 3) ndarray
                    RGB image with colorbar
    :param config:  string
                    Config option of the Tesseract model
    :return:        float
                    Temperature
    """
    temp_string = pytesseract.image_to_string(img, lang="eng", config=config)
    temp_string = re.split("[°C* \n]+", temp_string)[0]
    try:
        temp = float(temp_string)
    except ValueError as err:
        #         print(err)
        if err == "could not convert string to float: '§.0'":
            temp = 5.0
        else:
            temp = np.nan
    return temp
