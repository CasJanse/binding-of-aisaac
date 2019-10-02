import mss
import mss.tools
import numpy as np
import cv2
import time
import random
from PIL import Image


class ScreenshotMachine:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0

        # self.index = 0

        self.monitor = {}
        self.image_size = {}
        # self.output = ""

        self.key_index = 0
        self.resize_scale = 6

        self.set_coordinates(self.x, self.y, self.width, self.height)
        pass

    def take_screenshot(self):
        # self.index += 1
        with mss.mss() as sct:
            # output = "screenshots/sct-{top}x{left}_{width}x{height}_{index}.png".format(**{"left": self.x, "top": self.y, "width": self. width, "height": self.height, "index": self.index})

            # Take screenshot from monitor area
            sct_img = np.array(sct.grab(self.monitor))
            sct_img = cv2.resize(sct_img, dsize=(int(self.width / self.resize_scale), int(self.height / self.resize_scale)), interpolation=cv2.INTER_CUBIC)
            # sct_img = cv2.Canny(sct_img, 100, 200)

            # Save to the picture file
            # mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        return sct_img

    def set_coordinates(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.monitor = {"left": x, "top": y, "width": width, "height": height}
        self.image_size = {"x": int(width / self.resize_scale), "y": int(height / self.resize_scale)}
        pass

    def get_image_size(self):
        return self.image_size
        pass
