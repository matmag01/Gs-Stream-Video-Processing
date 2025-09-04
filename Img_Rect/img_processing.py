import cv2
import yaml
import argparse
import os
import sys
import numpy as np
import socket
import time
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

# Inizializza GStreamer
Gst.init(None)

def make_pipeline(device_number):
    """
    Create a GStreamer pipeline that captures video from a DeckLink device and sends it to an appsink.
    """
    pipeline_str = f"decklinkvideosrc device-number={device_number} ! videoconvert ! video/x-raw,format=BGR ! appsink name=appsink"
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("appsink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1)
    appsink.set_property("drop", True)
    return pipeline, appsink

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

def gst_to_opencv(sample):
    """
    Convert a GStreamer buffer into an OpenCV frame (numpy array)
    """
    buf = sample.get_buffer()
    caps = sample.get_caps()
    arr = np.ndarray(
        shape=(caps.get_structure(0).get_value("height"),
               caps.get_structure(0).get_value("width"),
               3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=np.uint8)
    return arr

def start_server(host="0.0.0.0", port=5000):
    """Start a TCP server and wait for client connection."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))     
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
    print("Camera matrix left:\n", cam_matrix_left)

    # Create two pipelines
    pipeline1, appsink1 = make_pipeline(0)
    pipeline1.set_state(Gst.State.PLAYING)
    pipeline2, appsink2 = make_pipeline(1)
    pipeline2.set_state(Gst.State.PLAYING)

    # Prepare rectification maps
    w, h = 1280, 720
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
        cap_time = time.time()
        sample1 = appsink1.emit("pull-sample")
        sample2 = appsink2.emit("pull-sample")
        if sample1 is None or sample2 is None:
            continue

        frame1 = gst_to_opencv(sample1)
        frame2 = gst_to_opencv(sample2)
        # Resize to 1280x720
        frame1 = cv2.resize(frame1, (1300, 1024), interpolation=cv2.INTER_LINEAR)
        frame2 = cv2.resize(frame2, (1300, 1024), interpolation=cv2.INTER_LINEAR)

        if frame2 is None:
            print("Error: Unable to read frame from camera 1")
            sys.exit(1)

        cv2.imshow("Left ", frame1)
        cv2.imshow("Right ", frame2)

        rec_time = time.time()
        print(f"Capture time: {rec_time - cap_time:.4f} seconds")
        # Rectification
        rect_left = cv2.remap(frame1, map1_left, map2_left, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame2, map1_right, map2_right, cv2.INTER_LINEAR)

        h, w = rect_left.shape[:2]
        print(f"Rectified resolution: {w}x{h}")
        
        encoding_time = time.time()
        print(f"Rectifying time: {encoding_time - rec_time:.4f} seconds")

        if args.view:
            cv2.imshow("Left Rectified", rect_left)
            cv2.imshow("Right Rectified", rect_right)

        if args.send and conn is not None:
            # Concatenate horizontally
            concat = np.hstack((rect_left, rect_right))
            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', concat, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])
            send_time = time.time()
            print(f"Encoding time: {send_time - encoding_time:.4f} seconds")
            data = np.array(buffer).tobytes()
            size = len(data)
            # Send size first (16 bytes, padded)
            conn.sendall(str(size).encode().ljust(16))
            # Then send the frame
            conn.sendall(data)
            print(f"Sending time: {time.time() - send_time:.4f} seconds")

        if (args.view and cv2.waitKey(1) & 0xFF == ord('q')):
            break

    pipeline1.set_state(Gst.State.NULL)
    pipeline2.set_state(Gst.State.NULL)
    if args.view:
        cv2.destroyAllWindows()
    if conn:
        conn.close()


if __name__ == "__main__":
    main()