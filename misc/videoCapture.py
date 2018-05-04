import cv2

cap = cv2.VideoCapture(1)
w=640
h=480
cap.set(3,w)
cap.set(4,h)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# frame size in out.write(frame) must be the same as the size argument in the constructor VideoWriter.
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (w,h))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # write the flipped frame
        out.write(frame)
        cv2.imshow('VideoStream', frame )
        if cv2.waitKey(1) == 27 :
            break 
    else:
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

