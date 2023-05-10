import cv2

# Function to record simple video of 4 seconds
# Args(camera_index; 1 = PC webcam, 1 = USB cam)
def record_vid(file_name, cam_index=0):
    # Set the video capture device (webcam)
    cap = cv2.VideoCapture(cam_index)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # Recording at 25 fps with Camera's original resolution
    out = cv2.VideoWriter(file_name, fourcc, 25.0, (width, height))

    # Set the duration of the video capture (in seconds)
    duration = 4

    # Capture frames for the specified duration
    start_time = cv2.getTickCount()
    frame_cnt = 0
    while (int((cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000) < duration * 1000):
        ret, frame = cap.read()

        if ret == True:
            # Show the frame
            cv2.imshow('frame', frame)
            # Write the frame to the output file
            out.write(frame)
            frame_cnt += 1
            # Wait for a key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_cnt == 86:
                break
        else:
            break

    # Release the video capture device and the output file
    cap.release()
    out.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":


    filename = ".\\Demonstration\\Demo_video.mp4"
    record_vid(filename)

    model = create_LRCN_model()

    print("Model_")

