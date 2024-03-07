import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the video url = https://vod-progressive.akamaized.net/exp=1709888873~acl=%2Fvimeo-prod-src-reg-us-east1%2Fvideos%2F2548958987~hmac=f5bc60b32ed18bcf920358840b1bf5b2dc6f5de426ccadc054156877bb90edd6/vimeo-prod-src-reg-us-east1/videos/2548958987?download=1&filename=pexels-ron-lach-7540501+%28Original%29.mp4&source=1
video = cv2.VideoCapture("pexels-ron-lach-7540501 (Original).mp4")

# Check if the video opened successfully
if not video.isOpened():
    print("Error opening video stream or file")

# Initialize tracker type
tracker_types = ['MIL', 'KCF', 'CSRT']
tracker_type = tracker_types[2]  # Using CSRT tracker

# Create a tracker based on tracker type
if tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL_create()
elif tracker_type == 'KCF':
    tracker = cv2.legacy.TrackerKCF_create()
elif tracker_type == "CSRT":
    tracker = cv2.legacy.TrackerCSRT_create()

# Read the first frame
ret, frame = video.read()

# Define an initial bounding box
x1, y1 = 2750, 1000, 
x2, y2 = 3050, 1250, 

bbox = (x1, y1, x2 - x1, y2 - y1)

# Initialize tracker with the first frame and bounding box
ok = tracker.init(frame, bbox)
scale_factor = 0.5  # Scale the frame to half; adjust this value as needed

# Tracking loop
while True:
    # Read a new frame
    ret, frame = video.read()
    if not ret:
        break  # Break the loop if there are no more frames
    
    # Update tracker
    ok, bbox = tracker.update(frame)

    # Resize frame for display
    display_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

    if ok:
        # Adjust bbox coordinates for the resized frame
        bbox_resized = (bbox[0] * scale_factor, bbox[1] * scale_factor, bbox[2] * scale_factor, bbox[3] * scale_factor)
        p1 = (int(bbox_resized[0]), int(bbox_resized[1]))
        p2 = (int(bbox_resized[0] + bbox_resized[2]), int(bbox_resized[1] + bbox_resized[3]))
        cv2.rectangle(display_frame, p1, p2, (0, 255, 0), 2, 1)
    else:
        # Tracking failure
        cv2.putText(display_frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75*(scale_factor), (0, 0, 255), 2)
    
    # Display result
    cv2.imshow("Tracking", display_frame)

    # Exit if ESC pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


# Compare the results:
# * Do you see any differences? If so, what are they?
# * Does one tracker perform better than the other? In what way?


# Так вижу отличия, что касается самой рамки то в одном случае она остается статична в друго может изменять свой размер
# В на моем пример лучше сработал MIL и CSRT. Хуже всего KCF, уже после 3 секунд видео он теряет объект и не находить его.  
