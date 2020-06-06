import cv2
import numpy as np
import time

# Create a VideoCapture object
p_camera = cv2.VideoCapture(2)
l_camera = cv2.VideoCapture(1)
# Check if camera opened successfully
if (p_camera.isOpened() == False):
    print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(p_camera.get(3))
frame_height = int(p_camera.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out1 = cv2.VideoWriter('prawa.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (frame_width, frame_height))
out2 = cv2.VideoWriter('lewa.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (frame_width, frame_height))
zapis=False
while (True):
    ret, frame = p_camera.read()
    ret1, frame1 = l_camera.read()
    if ret == True and ret1 == True:

        # Write the frame into the file 'output.avi'
        if cv2.waitKey(1) & 0xFF == ord('z'):
            zapis = True
        if zapis:
            time.sleep(0.3)
            out1.write(frame)
            out2.write(frame1)
            print ('Zapisywanie wideo')
        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('frame1', frame1)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

    # When everything done, release the video capture and video write objects
p_camera.release()
out1.release()
l_camera.release()
out2.release()

# Closes all the frames
cv2.destroyAllWindows()