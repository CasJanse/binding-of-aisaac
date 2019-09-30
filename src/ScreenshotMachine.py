import mss
import mss.tools
import numpy as np
import cv2
import time


class ScreenshotMachine:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0

        # self.index = 0

        self.monitor = {}
        # self.output = ""

        self.set_coordinates(self.x, self.y, self.width, self.height)
        pass

    def take_screenshots(self, amount_of_screenshots_per_second):
        while True:
            time.sleep(1 / amount_of_screenshots_per_second)
            # self.index += 1
            with mss.mss() as sct:
                # output = "screenshots/sct-{top}x{left}_{width}x{height}_{index}.png".format(**{"left": self.x, "top": self.y, "width": self. width, "height": self.height, "index": self.index})

                # Take screenshot from monitor area
                sct_img = np.array(sct.grab(self.monitor))

                cv2.imshow("OpenCV/Numpy grayscale", cv2.cvtColor(sct_img, cv2.COLOR_BGRA2GRAY))
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

                # Save to the picture file
                # mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
        pass

    def set_coordinates(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.monitor = {"left": x, "top": y, "width": width, "height": height}
        pass
