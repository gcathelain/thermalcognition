# -*- coding: utf-8 -*-
"""

This module contains functions for extracting a lamp temperature, that is switched on at stimulus start and off at
stimulus stop.

:copyright: (c) 2020 EPHE
:license: MIT License, see LICENSE for details
:author: Guillaume Cathelain
:organization: EPHE
:contact: guillaume.cathelain@gmail.com
:date: 23/12/2020
:version: 0.0
"""
import numpy as np


def get_lamp_temperature(videodata, crops=(0, 120, 0, 80)):
    """
    Finds lamp temperature, as the maximum of temperature in a specific region

    :param videodata:   (n, h, w) ndarray
                        Thermal video of the face
    :return:            (n,) ndarray
                        Maximum temperature of the lamp
    """
    n, h, w = videodata.shape
    top, bottom, left, right = crops
    lamp_mask = np.zeros((h, w), dtype=np.bool)
    lamp_mask[top:bottom, left:right] = True
    lamp_mask = lamp_mask.flatten()
    lamp_video_flat = videodata.reshape((n, -1))[:, lamp_mask]
    lamp_temperature = np.max(lamp_video_flat, axis=-1)
    return lamp_temperature
