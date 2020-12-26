# -*- coding: utf-8 -*-
"""

This module contains functions for reading and extracting face of thermal videos.

:copyright: (c) 2020 EPHE
:license: MIT License, see LICENSE for details
:author: Guillaume Cathelain
:organization: EPHE
:contact: guillaume.cathelain@gmail.com
:date: 22/12/2020
:version: 0.0
"""
import io
import subprocess
import fnv.file
import numpy as np
import pandas as pd
import cv2
from PIL.Image import Image
from tqdm import tqdm
from ffprobe import FFProbe
import skvideo.io
from .models import SFDDetector, transform
import spacy
from .calibration import rgb2temp_video, get_tempscale_df, fit_temperature, predict_temperature_video


def read_flir_video(video_path):
    """
    Reads FLIR .seq or .csq temperature video.

    :param video_path:  string
                        Path of the video
    :return:            (n, h, w) ndarray
                        The raw temperature video
    """
    video = fnv.file.ImagerFile(video_path)
    video.unit = fnv.Unit.TEMPERATURE_FACTORY
    n = video.num_frames
    height = video.height
    width = video.width
    videodata = np.zeros((n, height, width)) + np.nan
    for i in tqdm(range(n)):
        video.get_frame(i)
        videodata[i] = np.array(video.final, copy=False).reshape((height, width))
    return videodata


def read_flir_thermal_image(image_path, mode="FlirFileSDK"):
    """
    Reads FLIR .jpg temperature image.

    :param image_path:  string
                        Path of the .jpg image
    :param mode:        "FlirFileSDK" or "Exiftool"
                        Exiftool is experimental and gives bad results
    :return:            (h, w) ndarray
                        Temperature image
    """
    if mode == "FlirFileSDK":
        im = fnv.file.ImagerFile(image_path)
        im.unit = fnv.Unit.TEMPERATURE_FACTORY
        im.get_frame(0)
        image = np.array(im.final, copy=False).reshape((im.height, im.width))
    elif mode == "Exiftool":
        raw_thermal_image, _ = subprocess.Popen(["exiftool", "-b", "-FLIR:RawThermalImage", image_path],
                                                stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT).communicate()
        image = np.array(Image.open(io.BytesIO(raw_thermal_image)))
    else:
        raise NotImplementedError()
    return image


def read_flir_visible_image(image_path):
    """
    Reads FLIR .jpg embedded visible image, using Exiftool. Works with PiP or MSX mode.

    :param image_path:  string
                        Path of the .jpg image
    :return:            (h, w, 3) ndarray
                        Visible image, with 8 bits integers
    """
    visible_image, _ = subprocess.Popen(["exiftool", "-b", "-FLIR:EmbeddedImage", image_path],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT).communicate()
    image = np.array(Image.open(io.BytesIO(visible_image)))
    return image


def crop_face(image, crops=None, mode="thermal"):
    """
    Manually crops face of an image.

    :param image:   (h, w) ndarray
                    Contains the image to crop. Can also be a video.
    :param crops:   tuple of integers
                    Indices of left, top, right, bottom crops. None if mode is auto
    :param mode:    "thermal" or "visible"
                    The nature of the input image
    :return:        (h, w) ndarray
                    Image or video of the face
    """
    left, top, right, bottom = crops
    if mode == "thermal":
        if len(image.shape) > 2:
            image = image[:, top:bottom, left:right]
        else:
            image = image[top:bottom, left:right]
    elif mode == "visible":
        if len(image.shape) > 3:
            image = image[:, top:bottom, left:right]
        else:
            image = image[top:bottom, left:right]
    else:
        raise NotImplementedError()
    return image


def get_crops(image, path_to_detector, device="cuda"):
    """
    Detect faces automatically with a SFD detector.

    :param image:               (h, w, 3) ndarray
                                Contains the image to crop. Can also be a video.
    :param path_to_detector:    (h, w, 3) ndarray
                                Contains the image to crop. Can also be a video.
    :param device:              "cuda" or "cpu"
                                Whether to use GPU or CPU
    :return:                    tuple of integers
                                Indices of top, bottom, left, and right crops. None if mode is auto
    """
    detector = SFDDetector(device, path_to_detector=path_to_detector)
    bboxlist = detector.detect_from_image(image)[0]
    return bboxlist


def read_flir_image_metadata(image_path):
    """
    Reads FLIR image metadata using Exiftool.

    :param image_path:  string
                        Reads FLIR .jpg temperature image.
    :return:            pd.Dataframe
                        Contains Exiftool metadata tags, raw values and tokenize value and unit.
    """
    metadata, _ = subprocess.Popen(["exiftool", "-FLIR:all", image_path],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT).communicate()
    metadata = pd.read_csv(io.BytesIO(metadata), header=None, names=["Tag", "RawValue"], sep=":", error_bad_lines=False)
    metadata["Tag"] = metadata["Tag"].map(lambda x: x.replace(" ", ""))
    metadata["Value"] = metadata["RawValue"].map(lambda x: tokenize_unit_value(x)[0])
    metadata["Unit"] = metadata["RawValue"].map(lambda x: tokenize_unit_value(x)[1])
    return metadata


def tokenize_unit_value(raw_value):
    """
    Splits a raw value string into the related value and unit.

    :param raw_value:   string
    :return:            tuple
                        Contains (value, unit)
    """
    nlp = spacy.load("en_core_web_lg")
    tokens = list(nlp(raw_value))
    if len(tokens) == 1:
        value, unit = ("", "")
    elif len(tokens) == 2:
        if tokens[1].ent_type_ in ["CARDINAL", "QUANTITY", "DATE", ""]:
            value = tokens[1].text
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            unit = ""
        else:
            raise NotImplementedError(raw_value)
    elif len(tokens) == 3:
        if tokens[1].ent_type_ in ["CARDINAL", "QUANTITY", "PERCENT"]:
            value = tokens[1].text
            try:
                value = int(value)
            except ValueError:
                value = float(value)
            unit = tokens[2].text
        else:
            value = "".join([tokens[1].text, tokens[2].text])
            unit = ""
    elif len(tokens) == 4:
        if tokens[1].ent_type_ == tokens[2].ent_type_ == tokens[3].ent_type_ == "CARDINAL":
            value = [int(token.text) for token in tokens[1:]]
            unit = ""
        elif tokens[1].ent_type_ in ["DATE", "CARDINAL"]:
            value = "".join([token.text for token in tokens[1:]])
            unit = ""
        else:
            raise NotImplementedError(raw_value)
    else:
        value = " ".join([token.text for token in tokens[1:]])
        unit = ""
    return value, unit


def get_video_info(video_path):
    """
    Get the statistics of a video using FFProbe

    :param video_path:  string
                        The video path
    :return:            tuple
                        (duration, nb_frames, fps) of the video using FFProbe
    """
    video_reader = cv2.VideoCapture(video_path)
    nb_frames = -1
    ret = True
    while ret:
        ret, _ = video_reader.read()
        nb_frames += 1
    video_reader.release()
    duration = FFProbe(video_path).streams[0].duration_seconds()
    fps = nb_frames / duration
    return duration, nb_frames, fps


def read_testo_video(video_path, text_size=(11, 35), up_text_pos=(1, 325), low_text_pos=(-16, 325),
                     thermal_image_shape=(240, 320), dtype=np.float16, batch_size=1000):
    """
    Reads Testo .wmv thermal video.

    :param video_path:          string
                                The video path
    :param text_size:           (h, w) tuple
                                Shape of the box where to look for digits
    :param up_text_pos:         (y, x) tuple
                                Location of the box related to the upper temperature limit of the colorbar
    :param low_text_pos:        (y, x) tuple
                                Location of the box related to the lower temperature limit of the colorbar
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
    videodata = skvideo.io.vread(video_path)
    tempscale_df = get_tempscale_df(videodata, text_size=text_size, up_text_pos=up_text_pos, low_text_pos=low_text_pos)
    videodata = rgb2temp_video(videodata, tempscale_df, thermal_image_shape=thermal_image_shape, dtype=dtype,
                               batch_size=batch_size)
    return videodata


def read_testo_video_with_firstframe(video_path, xlsx_path, thermal_image_shape=(240, 320), batch_size=1000):
    """
    Fits the temperature and colors with the first frame Excel

    :param video_path:          string
                                The video path
    :param xlsx_path:           string
                                The temperature Excel path of the first frame
    :param thermal_image_shape: (h,w) tuple
                                Shape of the temperature image
    :param batch_size:          int
                                maximum number of frames used by iteration to avoid memory issues
    :return:                    (n, h, w) ndarray
                                Thermal frames of a video
    """
    videodata = skvideo.io.vread(video_path)
    h, w = thermal_image_shape
    videodata = videodata[:, :, :w]
    temperature_image = pd.read_excel(xlsx_path, engine="openpyxl", header=None).to_numpy()
    linreg_model = fit_temperature(videodata[0], temperature_image)
    videodata = predict_temperature_video(videodata, linreg_model, batch_size=batch_size)
    return videodata


def crop(image, center, scale, resolution=256.0):
    """
    Center crops an image or set of heatmaps

    :param image:       ndarray
                        an rgb image
    :param center:      ndarray
                        the center of the object, usually the same as of the bounding box
    :param scale:       float
                        scale of the face
    :param resolution:  float
                        the size of the output cropped image (default: {256.0})
    :return:            ndarray
                        an rgb image, cropped around the center
    """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg