import cv2
import sys
import subprocess
import numpy as np

def open_gst_cam(device_number):
    """Create a GStreamer pipeline to open the Blackmagic DeckLink camera."""
    pipeline_str = (
        f"decklinkvideosrc device-number={device_number} ! "
        "videoconvert ! "
        "videocrop left=310 right=310 top=28 bottom=28 ! "
        "video/x-raw,format=BGR ! "
        "appsink name=appsink"
    )
    cap = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {device_number}")
        sys.exit(1)
    return cap

def main():
    cap_left = open_gst_cam(0)

    # Prendiamo dimensioni dal primo frame
    ret, frame = cap_left.read()
    if not ret:
        print("Error: unable to grab first frame.")
        return
    height, width = frame.shape[:2]

    # Avvio ffmpeg in subprocess
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", "30",  # frame rate
        "-i", "-",   # input da stdin
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-f", "mpegts",
        "-muxdelay", "0.001",
        "http://0.0.0.0:8080/stream"
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print(f"Streaming {width}x{height} on http://<this-pc-ip>:8080/stream")

    try:
        while True:
            ret, frame = cap_left.read()
            if not ret:
                print("Error: Unable to read from video stream")
                break

            # Scrivi i bytes raw al pipe di ffmpeg
            process.stdin.write(frame.tobytes())

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap_left.release()
        process.stdin.close()
        process.wait()

if __name__ == "__main__":
    main()
