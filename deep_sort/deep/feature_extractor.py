from fastreid.utils.checkpoint import Checkpointer
from fastreid.modeling.meta_arch.build import build_model
from fastreid.config.config import get_cfg
from typing import Optional
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2
import logging

from .model import Net, BaseNet

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


class ResNet50Extractor(object):
    def __init__(self, use_cuda=True):
        self.net = BaseNet(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        logger = logging.getLogger("root.tracker")
        logger.info("Loading ResNet50 Model Pretrained on ImageNet")
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


class FastReIDExtractor(object):
    def __init__(self, fastreid_config_path: str, model_path: Optional[str], use_cuda=True):
        cfg = get_cfg()
        cfg.merge_from_file(fastreid_config_path)
        self.net = build_model(cfg)

        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        logger = logging.getLogger("root.tracker")
        self.net.to(self.device)
        self.net.eval()

        if model_path is not None:
            logger.info(f"Loading weights from {model_path}")
            Checkpointer(self.net).load(model_path)
        else:
            logger.info("Loading Model Pretrained on ImageNet")

        height, width = cfg.INPUT.SIZE_TEST
        self.size = (width, height)
        logger.info(f"Image size: {self.size}")
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):

        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        f = features.cpu().numpy()
        return f


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    print(cv2.imread("demo.jpg").shape)
    print(img.shape)
    # extr = Extractor("checkpoint/ckpt.t7")
    extr = ResNet50Extractor()
    im_crops = []
    im_crops.append(img)
    im_crops.append(img)
    feature = extr(im_crops)
    print(feature.shape)

