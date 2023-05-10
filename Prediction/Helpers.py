from collections import deque

import cv2
from lips import crop_lips
import mediapipe as mp
import numpy as np

w1_u = ['میں', 'آپ', 'ہم', 'وہ']
w2_k_u = ['کیسے', 'کونسا', 'کدھر', 'کتنے']
w3_u = ['ہوں', 'تھا', 'ہے', 'تھے']
w4_k_u = ['جی', 'ہاں', 'نہیں']
w5_u = ['کیوں', 'کب', 'کون']
w6_k_u = ['ایک', 'دو', 'تین', 'چار', 'پانچھ', 'چھے', 'سات', 'آٹھ', 'نوں']

word_list = [w1_u, w2_k_u, w3_u, w4_k_u, w5_u, w6_k_u]

# Important variables DEFINED IN PREPROCESSOR FILES
SEQUENCE_LENGTH = 15
IMAGE_HEIGHT = 50
IMAGE_WIDTH = 100
CLASSES_LIST = ['ہے', 'کیسے', 'چار', 'دو', 'چھے', 'وہ', 'جی', 'کب', 'پانچھ', 'تین', 'آپ', 'نوں', 'ہاں', 'میں', 'تھا', 'ہوں', 'نہیں', 'کیوں', 'کتنے', 'ایک', 'کون', 'تھے', 'ہم', 'آٹھ', 'کونسا', 'کدھر', 'سات']

# function to get most likely word in a given set of words
def get_most_likely_word(word_list, CLASSES_LIST, res):
    prob = float(0.0)
    pred = ''

    ls = []

    for i in word_list:
        index = CLASSES_LIST.index(i)
        tmp_prob = res[index]

        if tmp_prob > prob:
            prob = tmp_prob
            pred = i

    return pred, prob





# file_path = "..\\Dataset\\Urdu\\6\\aap_kitne_tha_han_kyun_ek\\_video.avi"

def predict_word_level(file_path, word_list, model, CLASSES_LIST, SEQUENCE_LENGTH):
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Declare a queue to store video frames.
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    sentence = []
    predictions = []
    threshold = 0.0  # Result rendered only if they are above this threshold

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    win_size = int((total_frames - 12) / 6)

    w_cnt = 0

    # Progress bar
    # f = IntProgress(min=0, max=total_frames-12) # instantiate the bar
    # display(f) # display the bar

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    preds = []

    w0 = []
    w1 = []
    w2 = []
    w3 = []
    w4 = []
    w5 = []

    cnt = 0
    with mp_holistic.Holistic(min_detection_confidence=0.1, min_tracking_confidence=0.1) as holistic:
        # Read until video is completed
        while (cap.isOpened()):

            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame correctly read only then performing predictions
            if ret == True:

                # Cropping lips
                cropped_image = crop_lips(frame, holistic)

                # Normalizing the cropped frame
                normalized_frame = cropped_image / 255

                # Appending the pre-processed frame into the frames list.
                frames_queue.append(normalized_frame)

                # Check if the number of frames in the queue are equal to the fixed sequence length.
                if len(frames_queue) == SEQUENCE_LENGTH:

                    # Pass the normalized frames to the model and get the predicted probabilities.
                    res = model.predict(np.expand_dims(frames_queue, axis=0), verbose=0)[0]

                    # print(CLASSES_LIST[np.argmax(res)])

                    # Appending prediction in the Predictions List
                    predictions.append(np.argmax(res))

                    # preds.append(get_most_likely_word(word_list[0], CLASSES_LIST, res))
                    # preds.append(get_most_likely_word(word_list[1], CLASSES_LIST, res))
                    # preds.append(get_most_likely_word(word_list[2], CLASSES_LIST, res))
                    # preds.append(get_most_likely_word(word_list[3], CLASSES_LIST, res))
                    # preds.append(get_most_likely_word(word_list[4], CLASSES_LIST, res))
                    # preds.append(get_most_likely_word(word_list[5], CLASSES_LIST, res))
                    # sentence.append(pred)

                    if w_cnt < win_size:
                        p0_w, p0_acc = get_most_likely_word(word_list[0], CLASSES_LIST, res)
                        if p0_acc > threshold:
                            w0.append([p0_w, p0_acc])

                    if w_cnt > win_size and w_cnt < (2 * win_size):
                        p1_w, p1_acc = get_most_likely_word(word_list[1], CLASSES_LIST, res)
                        if p1_acc > threshold:
                            w1.append([p1_w, p1_acc])

                    if w_cnt > (2 * win_size) and w_cnt < (3 * win_size):
                        p2_w, p2_acc = get_most_likely_word(word_list[2], CLASSES_LIST, res)
                        if p2_acc > threshold:
                            w2.append([p2_w, p2_acc])

                    if w_cnt > (3 * win_size) and w_cnt < (4 * win_size):
                        p3_w, p3_acc = get_most_likely_word(word_list[3], CLASSES_LIST, res)
                        if p3_acc > threshold:
                            w3.append([p3_w, p3_acc])

                    if w_cnt > (4 * win_size) and w_cnt < (5 * win_size):
                        p4_w, p4_acc = get_most_likely_word(word_list[4], CLASSES_LIST, res)
                        if p4_acc > threshold:
                            w4.append([p4_w, p4_acc])

                    if w_cnt > (5 * win_size):
                        p5_w, p5_acc = get_most_likely_word(word_list[5], CLASSES_LIST, res)
                        if p5_acc > threshold:
                            w5.append([p5_w, p5_acc])

                    w_cnt += 1

                    # Progress bar variables
                    # f.value += 1 # signal to increment the progress bar

                    # print(get_most_likely_word(word_list[5], CLASSES_LIST, res))
                    '''if np.unique(predictions[-5:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            print(CLASSES_LIST[np.argmax(res)])
                            if len(sentence) > 0:
                                if CLASSES_LIST[np.argmax(res)] != sentence[-1]:
                                    sentence.append(CLASSES_LIST[np.argmax(res)])
                            else:
                                sentence.append(CLASSES_LIST[np.argmax(res)])
                    '''

                    if len(sentence) < 8:
                        sentence.extend(np.unique(preds))

                # print(sentence)

                # cv2.putText(frame, ' '.join(sentence) , (2, 30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # Display the resulting frame
                # cv2.imshow('Frame', frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    return [w0, w1, w2, w3, w4, w5]


# Returns word with highest probability in a window
def highest_prob_word(words):
    w = ''
    acc = 0
    for i in words:
        if i[1] > acc:
            w = i[0]

    return w


# Returns most frequently occuring word in a window
def most_freq_word(words):
    tmp_list = []

    for i in words:
        tmp_list.append(i[0])

    return max(set(tmp_list), key=tmp_list.count)


# General function to parse sentence [args : prediction results, function to be used for getting predicted words]
def parse_sentence(prediction, func):
    sen = func(prediction[0]) + " " + func(prediction[1]) + " " + func(prediction[2]) + " " + func(
        prediction[3]) + " " + func(prediction[4]) + " " + func(prediction[5])

    return sen