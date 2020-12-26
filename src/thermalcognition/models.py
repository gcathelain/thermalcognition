# -*- coding: utf-8 -*-
"""

This module contains neural network functions for face and landmarks detection.

:copyright: (c) 2020 EPHE
:license: MIT License, see LICENSE for details
:author: Guillaume Cathelain
:organization: EPHE
:contact: guillaume.cathelain@gmail.com
:date: 23/12/2020
:version: 0.0
"""
import glob
import logging
import numpy as np
import cv2
from tqdm import tqdm
import torch
from skimage import io


class FAN(torch.nn.Module):
    """
    Face alignement network by Adrian Bulat
    """
    def __init__(self, num_modules=1):
        super(FAN, self).__init__()
        self.num_modules = num_modules

        # Base part
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), torch.nn.BatchNorm2d(256))
            self.add_module('l' + str(hg_module), torch.nn.Conv2d(256,
                                                            68, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), torch.nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), torch.nn.Conv2d(68,
                                                                 256, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)), True)
        x = torch.nn.functional.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = torch.nn.functional.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        return outputs


def get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.
    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]
    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True)

    return preds, preds_orig


def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.
    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.
    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution
    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()


class L2Norm(torch.nn.Module):
    """
    L2 Norm defined following PyTorch convention
    """

    def __init__(self, n_channels, scale=1.0):
        """
        Constructor method
        :param n_channels:
        :param scale:
        """
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.scale = scale
        self.eps = 1e-10
        self.weight = torch.nn.Parameter(torch.Tensor(self.n_channels))
        self.weight.data *= 0.0
        self.weight.data += self.scale

    def forward(self, x):
        """
        Formula
        :param x:
        :return:
        """
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x


class s3fd(torch.nn.Module):
    """
    Single Shot Scale-invariant Face Detector :  https://arxiv.org/abs/1708.05237
    Weights are located in the ./weights folder
    """

    def __init__(self):
        """
        Constructor method
        """
        super(s3fd, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc6 = torch.nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
        self.fc7 = torch.nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)
        self.conv6_1 = torch.nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv7_1 = torch.nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)
        self.conv3_3_norm_mbox_conf = torch.nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_3_norm_mbox_loc = torch.nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_conf = torch.nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv4_3_norm_mbox_loc = torch.nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_conf = torch.nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv5_3_norm_mbox_loc = torch.nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_conf = torch.nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.fc7_mbox_loc = torch.nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_conf = torch.nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
        self.conv6_2_mbox_loc = torch.nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_conf = torch.nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
        self.conv7_2_mbox_loc = torch.nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        Neural network model implementation
        :param x:   ndarray
                    Input image
        :return:
        """
        h = torch.nn.functional.relu(self.conv1_1(x))
        h = torch.nn.functional.relu(self.conv1_2(h))
        h = torch.nn.functional.max_pool2d(h, 2, 2)
        h = torch.nn.functional.relu(self.conv2_1(h))
        h = torch.nn.functional.relu(self.conv2_2(h))
        h = torch.nn.functional.max_pool2d(h, 2, 2)
        h = torch.nn.functional.relu(self.conv3_1(h))
        h = torch.nn.functional.relu(self.conv3_2(h))
        h = torch.nn.functional.relu(self.conv3_3(h))
        f3_3 = h
        h = torch.nn.functional.max_pool2d(h, 2, 2)
        h = torch.nn.functional.relu(self.conv4_1(h))
        h = torch.nn.functional.relu(self.conv4_2(h))
        h = torch.nn.functional.relu(self.conv4_3(h))
        f4_3 = h
        h = torch.nn.functional.max_pool2d(h, 2, 2)
        h = torch.nn.functional.relu(self.conv5_1(h))
        h = torch.nn.functional.relu(self.conv5_2(h))
        h = torch.nn.functional.relu(self.conv5_3(h))
        f5_3 = h
        h = torch.nn.functional.max_pool2d(h, 2, 2)
        h = torch.nn.functional.relu(self.fc6(h))
        h = torch.nn.functional.relu(self.fc7(h))
        ffc7 = h
        h = torch.nn.functional.relu(self.conv6_1(h))
        h = torch.nn.functional.relu(self.conv6_2(h))
        f6_2 = h
        h = torch.nn.functional.relu(self.conv7_1(h))
        h = torch.nn.functional.relu(self.conv7_2(h))
        f7_2 = h
        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)
        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = self.conv4_3_norm_mbox_conf(f4_3)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = self.conv5_3_norm_mbox_conf(f5_3)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = self.fc7_mbox_conf(ffc7)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = self.conv6_2_mbox_conf(f6_2)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = self.conv7_2_mbox_conf(f7_2)
        reg6 = self.conv7_2_mbox_loc(f7_2)
        # max-out background label
        chunk = torch.chunk(cls1, 4, 1)
        bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
        cls1 = torch.cat([bmax, chunk[3]], dim=1)
        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]


class FaceDetector(object):
    """An abstract class representing a face detector.
    Any other face detection implementation must subclass it. All subclasses
    must implement ``detect_from_image``, that return a list of detected
    bounding boxes. Optionally, for speed considerations detect from path is
    recommended.
    """

    def __init__(self, device, verbose):
        self.device = device
        self.verbose = verbose

        if verbose:
            if 'cpu' in device:
                logger = logging.getLogger(__name__)
                logger.warning("Detection running on CPU, this may be potentially slow.")

        if 'cpu' not in device and 'cuda' not in device:
            if verbose:
                logger.error("Expected values for device are: {cpu, cuda} but got: %s", device)
            raise ValueError

    def detect_from_image(self, tensor_or_path):
        """Detects faces in a given image.
        This function detects the faces present in a provided BGR(usually)
        image. The input can be either the image itself or the path to it.
        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- the path
            to an image or the image itself.
        Example::
            >>> path_to_image = 'data/image_01.jpg'
            ...   detected_faces = detect_from_image(path_to_image)
            [A list of bounding boxes (x1, y1, x2, y2)]
            >>> image = cv2.imread(path_to_image)
            ...   detected_faces = detect_from_image(image)
            [A list of bounding boxes (x1, y1, x2, y2)]
        """
        raise NotImplementedError

    def detect_from_batch(self, tensor):
        """Detects faces in a given image.
        This function detects the faces present in a provided BGR(usually)
        image. The input can be either the image itself or the path to it.

        Arguments:
            tensor {torch.tensor} -- image batch tensor.
        Example::
            >>> path_to_image = 'data/image_01.jpg'
            ...   detected_faces = detect_from_image(path_to_image)
            [A list of bounding boxes (x1, y1, x2, y2)]
            >>> image = cv2.imread(path_to_image)
            ...   detected_faces = detect_from_image(image)
            [A list of bounding boxes (x1, y1, x2, y2)]
        """
        raise NotImplementedError

    def detect_from_directory(self, path, extensions=['.jpg', '.png'], recursive=False, show_progress_bar=True):
        """Detects faces from all the images present in a given directory.
        Arguments:
            path {string} -- a string containing a path that points to the folder containing the images
        Keyword Arguments:
            extensions {list} -- list of string containing the extensions to be
            consider in the following format: ``.extension_name`` (default:
            {['.jpg', '.png']}) recursive {bool} -- option wherever to scan the
            folder recursively (default: {False}) show_progress_bar {bool} --
            display a progressbar (default: {True})
        Example:
        >>> directory = 'data'
        ...   detected_faces = detect_from_directory(directory)
        {A dictionary of [lists containing bounding boxes(x1, y1, x2, y2)]}
        """
        if self.verbose:
            logger = logging.getLogger(__name__)

        if len(extensions) == 0:
            if self.verbose:
                logger.error("Expected at list one extension, but none was received.")
            raise ValueError

        if self.verbose:
            logger.info("Constructing the list of images.")
        additional_pattern = '/**/*' if recursive else '/*'
        files = []
        for extension in extensions:
            files.extend(glob.glob(path + additional_pattern + extension, recursive=recursive))

        if self.verbose:
            logger.info("Finished searching for images. %s images found", len(files))
            logger.info("Preparing to run the detection.")

        predictions = {}
        for image_path in tqdm(files, disable=not show_progress_bar):
            if self.verbose:
                logger.info("Running the face detector on image: %s", image_path)
            predictions[image_path] = self.detect_from_image(image_path)

        if self.verbose:
            logger.info("The detector was successfully run on all %s images", len(files))

        return predictions

    @property
    def reference_scale(self):
        raise NotImplementedError

    @property
    def reference_x_shift(self):
        raise NotImplementedError

    @property
    def reference_y_shift(self):
        raise NotImplementedError

    @staticmethod
    def tensor_or_path_to_ndarray(tensor_or_path, rgb=True):
        """Convert path (represented as a string) or torch.tensor to a numpy.ndarray
        Arguments:
            tensor_or_path {numpy.ndarray, torch.tensor or string} -- path to the image, or the image itself
        """
        if isinstance(tensor_or_path, str):
            return cv2.imread(tensor_or_path) if not rgb else io.imread(tensor_or_path)
        elif torch.is_tensor(tensor_or_path):
            # Call cpu in case its coming from cuda
            return tensor_or_path.cpu().numpy()[..., ::-1].copy() if not rgb else tensor_or_path.cpu().numpy()
        elif isinstance(tensor_or_path, np.ndarray):
            return tensor_or_path[..., ::-1].copy() if not rgb else tensor_or_path
        else:
            raise TypeError


class SFDDetector(FaceDetector):
    def __init__(self, device, path_to_detector=None, verbose=False):
        super(SFDDetector, self).__init__(device, verbose)

        # Initialise the face detector
        model_weights = torch.load(path_to_detector)

        self.face_detector = s3fd()
        self.face_detector.load_state_dict(model_weights)
        self.face_detector.to(device)
        self.face_detector.eval()

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)

        bboxlist = detect(self.face_detector, image, device=self.device)[0]
        bboxlist = [x for x in bboxlist if x[-1] > 0.5]

        return bboxlist

    def detect_from_batch(self, tensor):
        bboxlists = batch_detect(self.face_detector, tensor, device=self.device)

        new_bboxlists = []
        for i in range(bboxlists.shape[0]):
            bboxlist = bboxlists[i]
            bboxlist = [x for x in bboxlist if x[-1] > 0.5]
            new_bboxlists.append(bboxlist)

        return new_bboxlists

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0


def detect(net, img, device):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    # Creates a batch of 1
    img = img.reshape((1,) + img.shape)

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    img = torch.from_numpy(img).float().to(device)

    return batch_detect(net, img, device)


def batch_detect(net, img_batch, device):
    """
    Inputs:
        - img_batch: a torch.Tensor of shape (Batch size, Channels, Height, Width)
    """

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    BB, CC, HH, WW = img_batch.size()

    with torch.no_grad():
        olist = net(img_batch.float())  # patched uint8_t overflow error

    for i in range(len(olist) // 2):
        olist[i * 2] = torch.nn.functional.softmax(olist[i * 2], dim=1)

    bboxlists = []

    olist = [oelem.data.cpu() for oelem in olist]

    for j in range(BB):
        bboxlist = []
        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            FB, FC, FH, FW = ocls.size()  # feature map size
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            anchor = stride * 4
            poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[j, 1, hindex, windex]
                loc = oreg[j, :, hindex, windex].contiguous().view(1, 4)
                priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])

        bboxlists.append(bboxlist)

    bboxlists = np.array(bboxlists)

    if 0 == len(bboxlists):
        bboxlists = np.zeros((1, 1, 5))

    return bboxlists


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = torch.nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = torch.nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = torch.nn.Sequential(
                torch.nn.BatchNorm2d(in_planes),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = torch.nn.functional.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = torch.nn.functional.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = torch.nn.functional.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class HourGlass(torch.nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = torch.nn.functional.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = torch.nn.functional.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)