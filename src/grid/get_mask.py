"""
O método implementado aqui tem como autor principal o mago Cassiano Klein

MaskCreator: Create mask for training using a simple object detection
method with CNN
apply_mask: Apply mask on the target img
debug_event: Debug event
"""

import os
import glob
from argparse import ArgumentParser

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def apply_mask(img: np.array, mask: np.array):
    """
    Apply mask on the target img

    Parameters
    ----------
    img: numpy array
        target image
    mask: numpy array
        mask

    Returns
    -------
    numpy array
        img with mask applied
    """

    img = img.copy()
    mask = cv.resize(mask, img.shape[:2], interpolation=cv.INTER_NEAREST)
    loc = np.where(mask == 1)
    img[loc[0], loc[1], 2] = 255
    loc = np.where(mask == 2)
    img[loc[0], loc[1], 0] = 255

    return img


def debug_event(event):
    """
    Debug event

    Parameters
    ----------
    event: any
        click event
    """

    print('Event %s' % event.name)
    print('Position (x, y) = (%s, %s)' % (event.xdata, event.ydata))
    print('Button %s' % event.button)


class MaskCreator:
    """
    Mask creator. Create mask for training using a simple object detection
    method with CNN
    """

    mask_sz = 15
    img_sz = 224
    factor = mask_sz / img_sz

    def __init__(self, path: str):
        self.path = path

        self.name, self.ext = os.path.splitext(os.path.basename(path))
        self.mpath = path.replace(self.ext, '_mask.bmp')
        if os.path.exists(self.mpath):
            self.mask = cv.imread(self.mpath)/60
        else:
            self.mask = np.zeros((self.mask_sz, self.mask_sz))

        self.img = cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB)
        self.img = cv.resize(self.img, (self.img_sz, self.img_sz),
                             interpolation=cv.INTER_CUBIC)
        self.fig, self.ax = plt.subplots()
        self.ax.grid(False)

        self.cur_class = 1
        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self.on_click)
        self.cid = self.fig.canvas.mpl_connect('button_release_event',
                                               self.on_release)
        self.cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update()

    def on_key(self, event):
        """
        Fill all  image with class 1

        Parameters
        ----------
        event: any
            matplotlib event
        """

        if event.key == 'a':
            self.mask[:, :] = 1
        self.update()

    def on_release(self, event):
        """
        Fill draw rectangle with current class

        Parameters
        ----------
        event: any
            matplotlib event
        """

        print('####################')
        debug_event(event)
        if event.button == 1:
            x, y = event.xdata, event.ydata
            print(x, y, self.factor)
            x, y = int((x * self.factor)), int((y * self.factor))
            print(x, y, self.factor)
            x0, x1 = min(self.downx, x), max(self.downx, x)
            y0, y1 = min(self.downy, y), max(self.downy, y)
            self.mask[y0:y1, x0:x1] = self.cur_class
            self.mask[y1, x0:x1] = self.cur_class
            self.mask[y0:y1, x1] = self.cur_class
            self.mask[y1, x1] = self.cur_class

        self.update()

    def on_click(self, event):
        """
        Choose event

        1 - left click
        2 - middle button
        3 - right click

        Parameters
        ----------
        event: any
            matplotlib event
        """

        print('####################')
        debug_event(event)
        if event.button == 3:  # right lcick
            self.cur_class = (self.cur_class + 1) % 3
        elif event.button == 1:
            x, y = event.xdata, event.ydata
            print(x, y, self.factor)
            x, y = int((x * self.factor)), int((y * self.factor))
            print(x, y, self.factor)
            self.downx, self.downy = x, y
        elif event.button == 2:
            cv.imwrite(self.mpath, self.mask*60)

    def update(self):
        """
        Update mask on the img
        """

        nimg = apply_mask(self.img, self.mask)

        self.ax.imshow(nimg)
        self.fig.canvas.draw()


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-p', '--path', help='File path')
    args = ap.parse_args()
    paths = glob.glob(args.path)
    for path in paths:
        p = MaskCreator(path)
        plt.show()
