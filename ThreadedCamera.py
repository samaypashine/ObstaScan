# /**
#  * @file ThreadedCamera.py
#  * @author Samay Pashine
#  * @brief Code to launch the camera feed in a seperate thread
#  * @version 1.0
#  * @date 2023-06-19
#  * 
#  * @copyright Copyright (c) 2023
#  * 
#  */

# Importing necessary libraries.
import cv2
import time
from threading import Thread

# Class to capture frames in different thread.
class ThreadedCamera(object):
    def __init__(self, source=0):

        self.capture = cv2.VideoCapture(source)
        time.sleep(2)
        self.thread = Thread(target=self.update, args=())

        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.capture.isOpened():
                self.capture.grab()
                time.sleep(0.005)

    def grab_frame(self):
        _, img = self.capture.retrieve()
        return img