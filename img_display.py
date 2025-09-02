import cv2
import yaml
import argparse
import os
import sys
import numpy as np
import socket
import time


def open_gst_cam(device_number):
    """Create a GStreamer pipeline to open the Blackmagic DeckLink camera."""
    gst_pipeline = (
        f"decklinkvideosrc device-number={device_number} ! video/x-raw width=1920,height=1080 ! deinterlace fields=top ! "
        "videoconvert ! appsink drop=true sync=false"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {device_number}")
        sys.exit(1)
    return cap

def main():

    # Open the two cameras
    cap_left = open_gst_cam(0)
    #cap_right = open_gst_cam(1)


    while True:
        # Capture frames
        retL, frameL = cap_left.read()
        retR = True #, frameR = cap_right.read()
        if not retL or not retR:
            print("Error: Unable to read from video streams")
            break

        cv2.imshow("Left", frameL)
        #cv2.imshow("Right", frameR)

        if (cv2.waitKey(2) & 0xFF == ord('q')):
            break

    cap_left.release()
    #cap_right.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
