import cv2
import numpy as np
import mediapipe as mp
import math

"""
    IMPORTANT VARIABLES: Width, Height of pre-processed frame
                         [Currently set to 100 & 50 respectively]
"""

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

# Passing image and model to the function
def mediapipe_detection(image, model):
    # Converting frame from BGR to RGB because model works on RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color conversion (BGR to RGB)

    image.flags.writeable = False  # Image not writeable anymore

    results = model.process(image)  # Making prediction

    image.flags.writeable = True  # Image is now writeable

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Color conversion (RGB to BGR)
    return image, results


def crop_lips(frame, model):
    image_height, image_width, c = frame.shape

    #IMPORTANT VARIABLES
    # Defining width and height of resized frame
    width = 100
    height = 50

    # If no lips get detected by mediapipe then exception will be thrown
    try:
        # Making detections

        image, results = mediapipe_detection(frame, model)


        # NORAMALIZING POSITIONS OF LANDMARKS(Two lines below taken from
        x_px1 = min(math.floor(results.face_landmarks.landmark[212].x * image_width), image_width - 1)
        x_px2 = min(math.floor(results.face_landmarks.landmark[432].x * image_width), image_width - 1)
        y_px1 = min(math.floor(results.face_landmarks.landmark[94].y * image_height), image_height - 1)
        y_px2 = min(math.floor(results.face_landmarks.landmark[200].y * image_height), image_height - 1)

        # Cropping an image
        cropped_image = image[y_px1:y_px2, x_px1:x_px2]

        # Resizing the cropped image to Fixed resolution i.e. 300*150
        dim = (width, height)

        resized = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)

    except:
        # If no lips detected plain black frame will be returned
        resized = np.zeros((height, width, 3), dtype=np.uint8)


    return resized




# Testing function on test image
if __name__ == "__main__":
    frame = cv2.imread("test_img.png", cv2.IMREAD_COLOR)

    # Accessing mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        cropped_image = crop_lips(frame, holistic)

        # Display cropped image
        cv2.imshow("cropped", cropped_image)

    cv2.waitKey(0)

    # It is for removing/deleting created GUI window from screen
    # and memory
    cv2.destroyAllWindows()

