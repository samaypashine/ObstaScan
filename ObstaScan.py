# /**
#  * @file ObstaScan.py
#  * @author Samay Pashine
#  * @brief Monocular distance estimation on just faces.
#  * @version 1.0
#  * @date 2023-06-19
#  * 
#  * @copyright Copyright (c) 2023
#  * 
#  */

# Importing the necessary library
import os
import cv2
import logging
from datetime import datetime
from ThreadedCamera import *

# Function to find the focal length of the camera.
def focal_length(measured_distance, real_width, width_in_rf_image):
    """
    brief:
        This Function Calculate the Focal Length (i.e. the distance between lens to CMOS sensor), it is simple constant we can find by using
        measured distance, read width (i.e the actual width of object) and width of the object in the image
    
    args:
        measure_distance (type: int): It is distance measured from object to the camera while capturing reference image
        real_width (type: int): It is actual width of object, in real world (like the width of the face used is = 14.3 cm)
        width_in_rf_image (type: int): It is object width in the image in our case in the ref. image
    
    return: 
        focal_length (type: float)
    """
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value

# Function to calculate distance.
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    """
    brief:
        Function simply estimates the distance between object and camera using focal length, actual object width, object width in the image.
    
    args:
        focal_length (type: float): focal length calculated with function.
        real_width (type: int): actual width of object in real world.
        object_width_frame (type: int): width of object in the image.
    
    return:
        distance (type: float): Estimated distance
    """
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance

# Function to detect face in the image and return list of widths
def face_detector(image):
    """
    brief:
        Function detects the faces in the images and return the image with bounding boxes and list of width
    
    args:
        image (type: Mat)
    
    return:
        list of widths
        image
    """
    face_width = []
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray_image, 1.3, 5)
    
    for (x, y, h, w) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN, 1)
        face_width.append(w)

    return image, face_width

# Driver code to initiate the application.
if __name__ == "__main__":
    # Configuring the log handler.
    log_path = r'logs' + os.sep + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    os.makedirs(r'logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
                        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])

    DEBUG = True
    cap = ThreadedCamera(0)

    # Distance from camera to object and Width of face in the real world measured in centimeter
    KNOWN_DISTANCE = 90
    KNOWN_WIDTH = 15
    
    # Colors code in BGR 
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)

    fonts = cv2.FONT_HERSHEY_COMPLEX
    detector = cv2.CascadeClassifier(os.sep.join(["haarcascade","haarcascade_frontalface_default.xml"]))

    # Calibration process using reference image.
    ref_image = cv2.imread("reference_image.jpg")
    ref_image, ref_image_face_width = face_detector(ref_image)
    focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width[0])
    logging.info(f'Focal length After Calibration : {focal_length_found}')
    cv2.imshow("Referece Image", ref_image)
    cv2.waitKey(0)

    while True:
        try:
            image = cap.grab_frame()
            image, image_face_width = face_detector(image)

            if len(image_face_width) != 0:
                Distance = distance_finder(focal_length_found, KNOWN_WIDTH, image_face_width[0])
                cv2.putText(image, f"Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, (RED), 2)
            
            cv2.imshow("Feed", image)
            if cv2.waitKey(1) == ord("q"):
                break

        except Exception as e:
            if DEBUG:
                logging.error(e)
            continue
    
    cv2.destroyAllWindows()
