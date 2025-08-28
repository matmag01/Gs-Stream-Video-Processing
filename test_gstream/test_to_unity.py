import cv2
import socket
import numpy as np

HOST = "0.0.0.0"  # server su tutte le interfacce
PORT = 5000

# GStreamer pipeline
pipeline = "videotestsrc ! videoconvert ! video/x-raw,format=BGR ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Server listening on: {HOST}:{PORT}")

# Accept connection on Unity
conn, addr = server_socket.accept()
print(f"Connection with {addr}")

if not cap.isOpened():
    print('Error: Video capture')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Video processing
    frame_modified = cv2.flip(frame, 1)
    # JPEG compression
    _, buffer = cv2.imencode('.jpg', frame_modified)
    data = np.array(buffer).tobytes()
    size = len(data)
    conn.sendall(str(size).encode().ljust(16))
    # Send data
    conn.sendall(data)

    # Show video
    cv2.imshow("Sender - modified", frame_modified)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()