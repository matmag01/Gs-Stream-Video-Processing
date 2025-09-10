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

# Inizializza GStreamer
Gst.init(None)

def make_capture_pipeline(device_number):
    """
    Crea una pipeline GStreamer per acquisire video e inviarlo a un appsink.
    """
    pipeline_str = (
        f"decklinkvideosrc device-number={device_number} ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink name=appsink"
    )
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("appsink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("max-buffers", 1)
    appsink.set_property("drop", True)
    return pipeline, appsink

def load_calibration(file_path):
    """Carica i parametri di calibrazione da un file YAML."""
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
    Converte un buffer GStreamer in un frame OpenCV (array numpy).
    """
    buf = sample.get_buffer()
    caps = sample.get_caps()
    height = caps.get_structure(0).get_value("height")
    width = caps.get_structure(0).get_value("width")
    
    arr = np.ndarray(
        shape=(height, width, 3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=np.uint8)
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml_dir", required=True, help="Directory contenente left.yaml e right.yaml")
    parser.add_argument("-v", "--view", action="store_true", help="Mostra i video rettificati")
    parser.add_argument("-p", "--publish", action="store_true", help="Pubblica i frame rettificati tramite GStreamer")
    parser.add_argument("--host", default="10.162.34.123", help="Host UDP (default: 10.162.34.123)")
    parser.add_argument("--port", type=int, default=7000, help="Porta UDP (default: 7000)")
    parser.add_argument("--framerate", type=int, default=24, help="Framerate di output (default: 24)")
    args = parser.parse_args()

    left_yaml = os.path.join(args.yaml_dir, "left.yaml")
    right_yaml = os.path.join(args.yaml_dir, "right.yaml")

    if not os.path.exists(left_yaml) or not os.path.exists(right_yaml):
        print("Errore: file left.yaml e/o right.yaml non trovati nella directory specificata.")
        sys.exit(1)

    cam_matrix_left, dist_left, rect_left, proj_left = load_calibration(left_yaml)
    cam_matrix_right, dist_right, rect_right, proj_right = load_calibration(right_yaml)

    pipeline1, appsink1 = make_capture_pipeline(0)
    pipeline1.set_state(Gst.State.PLAYING)
    pipeline2, appsink2 = make_capture_pipeline(1)
    pipeline2.set_state(Gst.State.PLAYING)

    w_out, h_out = 1300, 1024
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        cam_matrix_left, dist_left, rect_left, proj_left, (w_out, h_out), cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        cam_matrix_right, dist_right, rect_right, proj_right, (w_out, h_out), cv2.CV_16SC2)
    
    w_final = w_out * 2
    h_final = h_out
    
    out = None
    if args.publish:
        gst_out = (
            f"appsrc ! deinterlace ! videoconvert ! videorate ! "
            f"video/x-raw,framerate=24/1 ! x264enc name=videoEnc bitrate=1400 tune=zerolatency pass=qual ! "
            f"rtph264pay ! udpsink host={args.host} port={args.port} sync=false"
        )
        # La risoluzione e il framerate devono essere specificati qui
        out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, args.framerate, (w_final, h_final), True)
        if not out.isOpened():
            print("Errore: Impossibile aprire il VideoWriter. Controlla la pipeline GStreamer e le dipendenze.")
            sys.exit(1)

    try:
        while True:
            sample1 = appsink1.emit("pull-sample")
            sample2 = appsink2.emit("pull-sample")
            if sample1 is None or sample2 is None:
                continue

            frame1 = gst_to_opencv(sample1)
            frame2 = gst_to_opencv(sample2)

            rect_left = cv2.remap(frame1, map1_left, map2_left, cv2.INTER_LINEAR)
            rect_right = cv2.remap(frame2, map1_right, map2_right, cv2.INTER_LINEAR)
            
            concat_frame = np.hstack((rect_left, rect_right))
            
            if args.view:
                cv2.imshow("Combined Rectified", concat_frame)

            if args.publish and out:
                # Scrivi il frame processato nella pipeline di GStreamer
                out.write(concat_frame)
                print("Frame pubblicato")

            if (args.view and cv2.waitKey(1) & 0xFF == ord('q')):
                break
    finally:
        # Rilascia le risorse
        if out:
            out.release()
        pipeline1.set_state(Gst.State.NULL)
        pipeline2.set_state(Gst.State.NULL)
        if args.view:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
