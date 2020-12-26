# -*- coding: utf-8 -*-
"""

This module contains functions for extracting landmarks from thermal images.

:copyright: (c) 2020 EPHE
:license: MIT License, see LICENSE for details
:author: Guillaume Cathelain
:organization: EPHE
:contact: guillaume.cathelain@gmail.com
:date: 22/12/2020
:version: 0.0
"""
import os
import pandas as pd
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
from .reader import crop_face, get_video_info, crop
from .models import *


def load_fan(weights_path, device="cuda"):
    """
    Loads face alignment net from Adrian Bulat.
    Trained models come from "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar"

    :param weights_path:    string
                            Location of the compressed weights object
    :param device:          "cuda" or "cpu"
                            Whether use a GPU or a CPU
    :return:                torch nn
                            The face alignment neural network for landmarks detection
    """
    net = FAN(4)
    weights = torch.load(weights_path, map_location=torch.device(device))
    net.load_state_dict(weights)
    net.to(device)
    net.eval()
    return net


def preprocess_face(image, crops, net_resolution=256, device="cuda"):
    """
    Prepare face for landmark detection using FAN.

    :param image:           (h, w, 3) or (h, w) ndarray
                            Image of the face
    :param crops:           tuple of integers
                            Indices of top, bottom, left, and right crops
    :param net_resolution:  int
                            Size of the input layer of the neural network. Default for FAN
    :param device:          "cpu" or "cuda"
                            Whether to use GPU or CPU
    :return:                tuple
                            (h, w, 3) ndarray image of the face in RGB, center of the face, scale of the face
    """
    center = np.array([crops[2] - (crops[2] - crops[0]) / 2.0, crops[3] - (crops[3] - crops[1]) / 2.0])
    center[1] = center[1] - (crops[3] - crops[1]) * 0.12
    scale = (crops[2] - crops[0] + crops[3] - crops[1]) / 195  # reference scale
    if image.ndim == 2:
        height, width = image.shape
        image = image.reshape((height, width, 1))
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim == 4:
        image = image[..., :3]
    elif image.ndim == 3:
        pass
    else:
        raise NotImplementedError()
    image = crop(image, torch.FloatTensor(center), scale, resolution=net_resolution)
    inp = torch.from_numpy(image.transpose((2, 0, 1))).float()
    inp = inp.to(device)
    inp.div_(255.0).unsqueeze_(0)
    return inp, center, scale


def postprocess_landmarks(pts, crops=None, h=None, center=None, scale=None, net_resolution=256, mode="auto"):
    """
    Prepares superposition of landmarks on the original figure

    :param pts:             (68, 2) ndarray
                            Face landmarks and their row and column position
    :param crops:           tuple of integers
                            Indices of top, bottom, left, and right crops of the original image
    :param h:               int
                            height of the original image
    :param center:          ndarray
                            Center of the face
    :param scale:           float
                            Scale of the face
    :param mode:            "auto" or "manual"
                            Whether face detection is manual or automatic
    :param net_resolution:  int
                            Size of the input layer of the neural network. Default for FAN
    :return:                (68, 2) ndarray
                            Face landmarks and their row and column position in the original figure
    """
    if mode == "manual":
        top, bottom, left, right = crops
        pts = pts * h / net_resolution
        pts[:, 0] += left
    elif mode == "auto":
        # pts = np.array([np.array(transform(torch.from_numpy(pt.copy()), center, scale, resolution=256.)) for pt in pts])
        pts = np.array([np.array(transform(pt, center, scale, resolution=256.)) for pt in pts])
    else:
        raise NotImplementedError()
    return pts


def predict_landmarks(image, net):
    """
    Predicts landmarks of the face.

    :param image:   (h, w, 3) ndarray
                    Image of the face in RGB
    :param net:     torch nn
                    The FAN network for landmarks detection
    :return:        (68, 2) ndarray
                    Face landmarks and their row and column position
    """
    out = net(image)[-1].detach()
    out = out.cpu()
    pts, pts_img = get_preds_fromhm(out)
    pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
    return pts


def predict_landmarks_video(videodata, video_path, landmarks_detector_path, n_landmarks=68, decimate=1000, dpi=100,
                            device="cuda", save=False):
    """
    Predicts landmarks of the face video.

    :param videodata:               (n, h, w, 3) ndarray
                                    Visible video of the face
    :param video_path:              string
                                    The video path
    :param landmarks_detector_path: string
                                    Location of the compressed weights object
    :param n_landmarks:             int
                                    Number of landmarks detected by frame
    :param decimate:                int
                                    Decimation factor to shorten computation time
    :param dpi:                     int
                                    Resolution of output image
    :param device:                  "cuda" or "cpu"
                                    Whether use a GPU or a CPU
    :param save:                    bool
                                    Wether to save landmarks as mp4 and csv files
    :return:                        pd.DataFrame
                                    Contains landmarks position at every frame
    """
    landmarks_df = pd.DataFrame(
        data=np.nan + np.zeros((int(len(videodata) / decimate) + 1, 2 * n_landmarks + 1)),
        columns=["frame_index"] + list(np.array([["x" + str(i), "y" + str(i)] for i in range(n_landmarks)]).flatten()))
    _, h, w, _ = videodata.shape
    crops = ((-h + w) // 2, 0, (h - w) // 2, h)
    duration, nb_frames, fps = get_video_info(video_path)
    video_writer = FFMpegWriter(fps=fps / decimate)
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    axes_image = plt.imshow(np.random.randint(255, size=(h, w)), cmap="gray")
    scat = plt.scatter([], [])
    landmarks_net = load_fan(landmarks_detector_path, device=device)
    with video_writer.saving(fig, os.path.join(os.path.dirname(video_path), "landmarks.mp4"), dpi):
        for k, frame in tqdm(enumerate(videodata[::decimate])):
            face, center, scale = preprocess_face(frame, crops, device=device)
            pts = predict_landmarks(face, landmarks_net)
            pts = postprocess_landmarks(pts, center=center, scale=scale)
            if save:
                axes_image.set_data(frame.astype("uint8"))
                scat.set_offsets(pts)
                plt.draw()
                video_writer.grab_frame()
            landmarks_df.iloc[k, 0] = int(k * decimate)
            landmarks_df.iloc[k, 1:] = pts.flatten()
    plt.close()
    landmarks_df = landmarks_df.dropna(axis="index", how="any")
    landmarks_df["frame_index"] = landmarks_df["frame_index"].astype("int")
    if save:
        landmarks_df.to_csv(os.path.join(os.path.dirname(video_path), "landmarks.csv"))
    return landmarks_df
