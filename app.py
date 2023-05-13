import cv2
import time
from flask import Flask, render_template, Response, request
from LRCN import create_LRCN_model
from record_video import record_vid
from Helpers import predict_word_level, word_list, CLASSES_LIST, SEQUENCE_LENGTH, parse_sentence, most_freq_word, highest_prob_word


app = Flask(__name__)
video_capture = cv2.VideoCapture(0)  # 0 represents the default webcam
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_filename = 'recorded_video.mp4'
recording = False
out = None

start_time = 0


def generate_frames():
    global recording,start_time

    while True:
        success, frame = video_capture.read()  # Read frames from the webcam
        if not success:
            break

        if time.time() - start_time >= 4:
            stop_recording()


        # if recording == true save frame to video
        if recording:
            out.write(frame)  # Write the frame to the video file


        ret, buffer = cv2.imencode('.jpg', frame)

        #buffer = cv2.flip(buffer, 0)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Generate video frame as JPEG data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording')
def start_recording():
    global recording, out, start_time

    if not recording:
        start_time = time.time()
        recording = True
        out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'DIVX'), 25.0, (width, height))

    return "Recording started"


@app.route('/stop_recording')
def stop_recording():
    global recording, out


    if recording:
        recording = False
        out.release()
        #str = predict_video()
    return "Recording stopped"

@app.route('/predict_video')
def predict_video():
    global output_filename

    output_filename = "./Demonstration/Demo.mp4"

    #if recording:
    #    recording = False
    #    out.release()

    weights_path = "./Demonstration/Model_1-Without-CTC-LOSS_checkpoint1_tillbatch_9.h5"
    model = create_LRCN_model()
    model.load_weights(weights_path)

    print("\nModel predicting...")
    prediction = predict_word_level(output_filename, word_list, model, CLASSES_LIST, SEQUENCE_LENGTH)

    # print(parse_sentence(prediction, highest_prob_word))
    print(parse_sentence(prediction, most_freq_word))

    return "Prediction Completed"

if __name__ == '__main__':
    app.run(debug=True)
