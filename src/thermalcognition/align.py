# -*- coding: utf-8 -*-
"""

This module contains functions for aligning thermal and visible images. Works with PiP and MSX mode of FLIR images.

:copyright: (c) 2020 EPHE
:license: MIT License, see LICENSE for details
:author: Guillaume Cathelain
:organization: EPHE
:contact: guillaume.cathelain@gmail.com
:date: 22/12/2020
:version: 0.0
"""
import cv2
import numpy as np
from PIL.Image import Image
import os
from matplotlib import cm
from .reader import read_flir_thermal_image, read_flir_visible_image, read_flir_image_metadata, crop_face


def align_flir_visible_thermal(image_path, save=False, colorscale="plasma"):
    """
    Aligns visible and thermal FLIR images in PiP or MSX mode.

    :param image_path:  string
                        Path of the .jpg FLIR image, containing thermal and visible images.
    :param save:        bool
                        Save metadata.csv, visible.jpg, thermal.jpg and flir.jpg images
    :param colorscale:  string
                        Colorscale to use when saving thermal images. Plasma is near the Flir Iron corlorscale
    :return:            tuple
                        (thermal, visible) images
    """
    thermal_image = read_flir_thermal_image(image_path)
    visible_image = read_flir_visible_image(image_path)
    metadata = read_flir_image_metadata(image_path)
    if save:
        metadata_path = os.path.join(os.path.dirname(image_path), "metadata.csv")
        metadata.to_csv(metadata_path)
    Real2IR, OffsetX, OffsetY = [
        metadata.loc[metadata["Tag"] == tag, "Value"].squeeze() for tag in ["Real2IR", "OffsetX", "OffsetY"]]
    cropY, cropX = (np.array(visible_image.shape[:2]) * (1 - 1 / Real2IR)).astype("int")
    if -cropY // 2 + OffsetY == 0:
        pass
    elif -cropY // 2 + OffsetY < 0:
        visible_image = visible_image[cropY // 2 + OffsetY:-cropY // 2 + OffsetY, :]
    else:
        raise NotImplementedError()
    if -cropX // 2 + OffsetX == 0:
        pass
    elif -cropX // 2 + OffsetX < 0:
        visible_image = visible_image[:, cropX // 2 + OffsetX: -cropX // 2 + OffsetX]
    else:
        raise NotImplementedError()
    _ = get_ratio(visible_image, thermal_image)
    visible_image = cv2.resize(visible_image, tuple(reversed(thermal_image.shape[:2])), interpolation=cv2.INTER_AREA)
    if save:
        flir_image = np.array(Image.open(image_path))
        flir_image = cv2.resize(flir_image, tuple(reversed(thermal_image.shape)), interpolation=cv2.INTER_AREA)
        plasma = cm.get_cmap(colorscale, 255)
        thermal_image = plasma(
            (thermal_image - np.min(thermal_image)) / (np.max(thermal_image) - np.min(thermal_image)))[:, :, :3]
        thermal_image = (thermal_image * 255).astype("uint8")
        visible_path = os.path.join(os.path.dirname(image_path), "visible.jpg")
        thermal_path = os.path.join(os.path.dirname(image_path), "thermal.jpg")
        flir_path = os.path.join(os.path.dirname(image_path), "flir.jpg")
        Image.fromarray(visible_image).save(visible_path)
        Image.fromarray(thermal_image).save(thermal_path)
        Image.fromarray(flir_image).save(flir_path)
    return thermal_image, visible_image


def build_flir_pip(image_path, save=False):
    """
    Reconstruct PiP image from scratch

    :param image_path:  string
                        Flir PiP image
    :param save:        bool
                        Whether to save the generated PiP image
    :return:            (h, w, 3) array
                        Reconstructed flir PiP image
    """
    pip_image = Image.open(image_path)
    pip_image = np.array(pip_image)
    visible_image = read_flir_visible_image(image_path)
    thermal_image = read_flir_thermal_image(image_path)
    metadata = read_flir_image_metadata(image_path)
    Real2IR, OffsetX, OffsetY, PiPX1, PiPX2, PiPY1, PiPY2 = [
        metadata.loc[metadata["Tag"] == tag, "Value"].squeeze()
        for tag in ["Real2IR", "OffsetX", "OffsetY", "PiPX1", "PiPX2", "PiPY1", "PiPY2"]
    ]
    cropY, cropX = (np.array(visible_image.shape[:2]) * (1 - 1 / Real2IR)).astype("int")
    built_pip_image = visible_image[cropY // 2 + OffsetY:-cropY // 2 + OffsetY,
                      cropX // 2 + OffsetX: -cropX // 2 + OffsetX]
    built_pip_image = cv2.resize(built_pip_image, tuple(reversed(pip_image.shape[:2])),
                                 interpolation=cv2.INTER_AREA)
    _ = get_ratio(visible_image, thermal_image)
    _ = get_ratio(visible_image, pip_image)
    ratio = get_ratio(pip_image, thermal_image)
    pt_image = thermal_image[PiPY1: PiPY2, PiPX1:PiPX2]
    new_pt_image_shape = tuple(np.flip(np.array(pt_image.shape[:2]) * ratio).astype("int"))
    pt_image = cv2.resize(pt_image, new_pt_image_shape, interpolation=cv2.INTER_AREA)
    plasma = cm.get_cmap('plasma', 255)
    pt_image = plasma((pt_image - np.min(pt_image)) / (np.max(pt_image) - np.min(pt_image)))
    pt_image = pt_image[:, :, :3]
    pip_ir_image = (pt_image * 255).astype("int")
    cropY, cropX = np.array(built_pip_image.shape[:2]) - np.array(pip_ir_image.shape[:2])
    built_pip_image[cropY // 2:-cropY // 2, cropX // 2:-cropX // 2] = pip_ir_image
    if save:
        built_pip_path = os.path.join(os.path.dirname(image_path), "built_pip.jpg")
        Image.fromarray(built_pip_image).save(built_pip_path)
    return built_pip_image


def get_ratio(image1, image2):
    """
    Get the scale ratio between row and column dimensions of two images. If height and width dimensions are different,
    throws an error.

    :param image1:  (h, w, 3) or (h, w) array
                    Image to be compared
    :param image2:  (h, w, 3) or (h, w) array
                    Reference image
    :return:        float
                    Ratio between image1 and the reference image
    """
    if np.diff(np.array(image1.shape[:2]) / np.array(image2.shape[:2])) == 0:
        ratio = image1.shape[0] / image2.shape[0]
    else:
        raise ValueError("different height and width ratio")
    return ratio


def face_template_matching(visible_image, thermal_image, bboxlist):
    """
    Align visible and thermal face imaging using template matching. Used in VIS-TH database.

    :param visible_image:   (h, w, 3) ndarray
                            Visible image on which face was detected
    :param thermal_image:   (h, w) ndarray
                            Thermal image to align
    :param bboxlist:        tuple
                            x1, y1, x2, y2, confidence boundaries of the visible face box
    :param net_resolution:  int
                            Size of the input layer of the neural network. Default for FAN
    :param device:          "cpu" or "cuda"
                            Whether to use GPU or CPU
    :return:                tuple
                            visible_face and thermal_face
    """
    x1, y1, x2, y2, confidence = bboxlist
    template = np.mean(
        visible_image[int(bboxlist[1]):int(bboxlist[3]), int(bboxlist[0]):int(bboxlist[2]), :], axis=-1)
    image = thermal_image[int(bboxlist[1]):int(bboxlist[3]), :]
    corr_coeff = cv2.matchTemplate(image.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    corr_coeff = np.squeeze(corr_coeff)
    delta_x = np.argmax(corr_coeff) - x1
    visible_crops = np.array(bboxlist[:4], dtype="int")
    thermal_crops = np.array([x1 + delta_x, y1, x2 + delta_x, y2], dtype="int")
    visible_face = crop_face(visible_image, visible_crops, mode="visible")
    thermal_face = crop_face(thermal_image, thermal_crops, mode="thermal")
    return visible_face, thermal_face


def resize_visible_flirone(visible_image, thermal_image):
    """
    Resize visible image using thermal image. Used in VIS-TH database.

    :param visible_image:   (h, w, 3) ndarray
                            Visible image on which face was detected
    :param thermal_image:   (h, w) ndarray
                            Thermal image to align
    :return:                tuple
                            visible_face and thermal_face
    """
    h_visible, w_visible, _ = visible_image.shape
    h_thermal, w_thermal = thermal_image.shape
    scale = h_visible / h_thermal  # 9.0
    scale = int(scale)
    crop_width = w_visible - w_thermal * scale
    visible_image = visible_image[:, crop_width // 2:-crop_width // 2, :]
    visible_image = cv2.resize(visible_image, (w_thermal, h_thermal), interpolation=cv2.INTER_AREA)
    return visible_image
