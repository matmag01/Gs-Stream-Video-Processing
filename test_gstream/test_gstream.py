import cv2
import time
# Import video from gstream

# 1) pipeline
pipeline = "videotestsrc ! videoconvert ! video/x-raw,format=BGR ! appsink"
# 2) Open the video with opencv
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
# 3) Check if correctly captured
if not cap.isOpened():
    print('Error in video capture: check pipeline dir')
    exit()
else:
    # Video settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Set as default
    print('Video correctly captured')

# Out
gst_out = (
    "appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=ultrafast "
    "! rtph264pay config-interval=1 pt=96 ! udpsink host=10.162.34.96 port=5000"
)
out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, fps, (width, height), True)

while True:
# Read from pipeline
    ret, frame = cap.read()
    # Check if frame is successfully read
    if not ret:
        print('Unable to read file')
        break

    # Timestamp
    ts = time.time()
    cv2.putText(frame, f"{ts:.6f}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Display
    cv2.imshow('Frame', frame)
    frame_modified = cv2.flip(frame, 1) # Example: flip image
    cv2.imshow("Sender - modified (flipped)", frame_modified)
    out.write(frame)
    # Check if the user pressed 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()


