import cv2
import yaml
import argparse
import os
import sys
import numpy as np
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp

# Initialize GStreamer
Gst.init(None)

def make_capture_pipeline(device_number):
    """
    Create a GStreamer pipeline to capture video and send it to an appsink (appsink is a sink plugin that supports 
    many different methods for making the application get a handle on the GStreamer data in a pipeline).
    """

    # Pipeline string
    pipeline_str = (
        f"decklinkvideosrc device-number={device_number} ! deinterlace !"
        "videoconvert ! "
        "videocrop left=310 right=310 top=28 bottom=28 ! " # Crop to 1300x1024
        "video/x-raw,format=BGR ! " # Convert to BGR format for OpenCV
        "appsink name=appsink"
    )

    # Pipeline GStreamer creation
    pipeline = Gst.parse_launch(pipeline_str)
    # appsink properties
    appsink = pipeline.get_by_name("appsink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1) # Keep only the latest frame
    appsink.set_property("drop", True) # Drop old frames
    return pipeline, appsink

def load_calibration(file_path):
    """Upload calibration parameters from a YAML file."""
    with open(file_path, 'r') as f:
        calib = yaml.safe_load(f)

    def to_np(mat):
        return np.array(mat["data"], dtype=np.float32).reshape((mat["rows"], mat["cols"]))

    camera_matrix = to_np(calib["camera_matrix"])
    dist_coeffs = np.array(calib["distortion_coefficients"]["data"], dtype=np.float32)
    rect_matrix = to_np(calib["rectification_matrix"])
    proj_matrix = to_np(calib["projection_matrix"])

    return camera_matrix, dist_coeffs, rect_matrix, proj_matrix

def gst_to_opencv(sample):
    """
    From GStreamer sample (buffer) to OpenCV image (numpy array).
    """
    buf = sample.get_buffer() # Get the buffer from the sample
    caps = sample.get_caps() # Get the capabilities/properties of the sample
    height = caps.get_structure(0).get_value("height")
    width = caps.get_structure(0).get_value("width")
    
    arr = np.ndarray(
        shape=(height, width, 3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=np.uint8)
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml_dir", required=True, help="Directory containing left.yaml and right.yaml")
    parser.add_argument("-v", "--view", action="store_true", help="Show rectified videos")
    parser.add_argument("-p", "--publish", action="store_true", help="Publish rectified frames via UDP")
    parser.add_argument("--host", default="10.162.34.123", help="Host UDP (default: 10.162.34.123)")
    parser.add_argument("--port", type=int, default=7000, help="Port UDP (default: 7000)")
    parser.add_argument("--framerate", type=int, default=24, help="Output framerate (default: 24)")
    args = parser.parse_args()

    # Load calibration files
    left_yaml = os.path.join(args.yaml_dir, "left.yaml")
    right_yaml = os.path.join(args.yaml_dir, "right.yaml")

    if not os.path.exists(left_yaml) or not os.path.exists(right_yaml):
        print("Error: left.yaml and/or right.yaml not found in the specified directory.")
        sys.exit(1)

    cam_matrix_left, dist_left, rect_left, proj_left = load_calibration(left_yaml)
    cam_matrix_right, dist_right, rect_right, proj_right = load_calibration(right_yaml)

    # Create capture pipelines for both cameras
    pipeline1, appsink1 = make_capture_pipeline(0)
    pipeline1.set_state(Gst.State.PLAYING)
    pipeline2, appsink2 = make_capture_pipeline(1)
    pipeline2.set_state(Gst.State.PLAYING)

    # Prepare rectification maps
    w_out, h_out = 1300, 1024
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        cam_matrix_left, dist_left, rect_left, proj_left, (w_out, h_out), cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        cam_matrix_right, dist_right, rect_right, proj_right, (w_out, h_out), cv2.CV_16SC2)
    
    w_final = w_out * 2 # Concatenate side by side
    h_final = h_out
    
    out = None

    if args.publish:
        gst_out = (
            f"appsrc ! deinterlace ! videoconvert ! videorate ! "
            f"video/x-raw,framerate=24/1 ! x264enc name=videoEnc bitrate=1480 tune=zerolatency pass=qual ! "
            f"rtph264pay ! udpsink host={args.host} port={args.port} sync=false"
        )
        print("Starting GStreamer pipeline for publishing...")
        print(gst_out)
        
        out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, args.framerate, (w_final, h_final), True)
        if not out.isOpened():
            print("Error: Unable to open VideoWriter. Check GStreamer pipeline and dependencies.")
            sys.exit(1)

    try:
        while True:
            # Pull samples from appsinks
            sample1 = appsink1.emit("pull-sample")
            sample2 = appsink2.emit("pull-sample")
            if sample1 is None or sample2 is None:
                continue

            frame1 = gst_to_opencv(sample1)
            frame2 = gst_to_opencv(sample2)

            # Rectification
            rect_left = cv2.remap(frame1, map1_left, map2_left, cv2.INTER_LINEAR)
            rect_right = cv2.remap(frame2, map1_right, map2_right, cv2.INTER_LINEAR)

            # Concatenation
            concat_frame = np.hstack((rect_left, rect_right))
            
            if args.view:
                cv2.imshow("Combined Rectified", concat_frame)

            if args.publish and out:
                # Write the processed frame to the GStreamer pipeline
                out.write(concat_frame)
                #print("Frame published")

            if (args.view and cv2.waitKey(1) & 0xFF == ord('q')):
                break
    finally:
        # Cleanup
        if out:
            out.release()
        pipeline1.set_state(Gst.State.NULL)
        pipeline2.set_state(Gst.State.NULL)
        if args.view:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()