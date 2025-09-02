import cv2
import yaml
import argparse
import os
import sys
import numpy as np
import socket
import time

def load_calibration(file_path):
    """Load calibration parameters from a YAML file."""
    with open(file_path, 'r') as f:
        calib = yaml.safe_load(f)

    # Convert blocks to numpy arrays
    def to_np(mat):
        return np.array(mat["data"], dtype=np.float32).reshape((mat["rows"], mat["cols"]))

    camera_matrix = to_np(calib["camera_matrix"])
    dist_coeffs   = np.array(calib["distortion_coefficients"]["data"], dtype=np.float32)
    rect_matrix   = to_np(calib["rectification_matrix"])
    proj_matrix   = to_np(calib["projection_matrix"])

    return camera_matrix, dist_coeffs, rect_matrix, proj_matrix

def open_gst_cam(device_number):
    """Create a GStreamer pipeline to open the Blackmagic DeckLink camera."""
    gst_pipeline = (
        f"decklinkvideosrc device-number={device_number} ! "
        "videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {device_number}")
        sys.exit(1)
    return cap

def start_server(host="0.0.0.0", port=5000):
    """Start a TCP server and wait for client connection."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))       # <-- QUI invece di connect()
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}")
    
    conn, addr = server_socket.accept()
    print(f"Connection with {addr}")
    return conn

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("-y", "--yaml_dir", required=True,
                        help="Directory containing left.yaml and right.yaml")
    parser.add_argument("-v", "--view", action="store_true",
                        help="Show rectified videos")
    parser.add_argument("-s", "--send", action="store_true",
                        help="Concatenate and send rectified frames via TCP")
    parser.add_argument("--host", default="0.0.0.0",
                        help="TCP host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000,
                        help="TCP port (default: 5000)")
    parser.add_argument("-q", "--quality", type=int, default=100,
                        help="JPEG quality (default: 100)")
    args = parser.parse_args()

    # YAML file paths
    left_yaml = os.path.join(args.yaml_dir, "left.yaml")
    right_yaml = os.path.join(args.yaml_dir, "right.yaml")

    if not os.path.exists(left_yaml) or not os.path.exists(right_yaml):
        print("Error: left.yaml and/or right.yaml files not found in the specified directory")
        sys.exit(1)

    # Load parameters
    cam_matrix_left, dist_left, rect_left, proj_left = load_calibration(left_yaml)
    cam_matrix_right, dist_right, rect_right, proj_right = load_calibration(right_yaml)
    # Open the two cameras
    cap_left = open_gst_cam(0)
    cap_right = open_gst_cam(1)

    # Get dimensions
    ret, frame = cap_left.read()
    if not ret:
        print("Error: Unable to read frame from camera 0")
        sys.exit(1)
    h, w = frame.shape[:2]

    # Prepare rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        cam_matrix_left, dist_left, rect_left, proj_left, (w, h), cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        cam_matrix_right, dist_right, rect_right, proj_right, (w, h), cv2.CV_16SC2)
    
    # Start TCP server if requested
    conn = None
    if args.send:
        conn = start_server(args.host, args.port)

    while True:
        # Capture frames
        retL, frameL = cap_left.read()
        retR, frameR = cap_right.read()
        if not retL or not retR:
            print("Error: Unable to read from video streams")
            break
        time_rec = time.time()
        # Rectification
        rect_left = cv2.remap(frameL, map1_left, map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frameR, map1_right, map2_right, cv2.INTER_LINEAR)
        print(f"Time to rectify: {time.time() - time_rec:.4f} seconds")

        if args.view:
            cv2.imshow("Left Rectified", rect_left)
            cv2.imshow("Right Rectified", rect_right)

        if args.send and conn is not None:
            # Concatenate horizontally
            concat = np.hstack((rect_left, rect_right))
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', concat, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            data = np.array(buffer).tobytes()
            size = len(data)
            # Send size first (16 bytes, padded)
            conn.sendall(str(size).encode().ljust(16))
            # Then send the frame
            conn.sendall(data)

        if (args.view and cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap_left.release()
    cap_right.release()
    if args.view:
        cv2.destroyAllWindows()
    if conn:
        conn.close()


if __name__ == "__main__":
    main()
