import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging

from .model import Net

'''
Feature extractor:
Extracts features from the corresponding bounding box and returns a fixed-size
embedding as the bounding box representation for similarity computation.

The model is trained with a traditional ReID workflow. The Extractor class takes
a list of images as input and returns the corresponding image features.
'''

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
            # RGB image values are in [0, 255]. ToTensor first divides by 255 to
            # normalize them to [0, 1], then Normalize applies (x - mean) / std
            # to scale the data to [-1, 1].
            transforms.ToTensor(),
            # The mean and std values are computed from the ImageNet training set.
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

# __call__() lets the class instance be invoked like a regular function,
# similar to overloading the () operator.
    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("../../../.weights/.deep_sort_checkpoint/ckpt_person.t7")
    feature = extr(img)
    print(feature.shape)

