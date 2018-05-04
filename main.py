import cv2
import numpy as np


class camera:
    'status about the captured frame'
    deviceId = 1
    dbg = 0
    currentFrame = []
    lastFrame = []
    dbgFrame = []
    def __init__(self, dbg=0):
        self.video = cv2.VideoCapture(self.deviceId)
        self.dbg = dbg
    def getFrame(self):
        ret, frame  = self.video.read()
        if ret:
            self.lastFrame = self.currentFrame
            self.currentFrame = frame
            self.dbgFrame = np.zeros(frame.shape,np.uint8)
            return 1
        return 0
    def lightIntensity(self, frame):
        # Convert
        hsv = cv2.cvtColor( frame, cv2.COLOR_BGR2HSV)
        v_mean = np.mean(hsv[2])
        return v_mean

class scanner_object:
    'known object for the scanner to recognize'
    objectId = 0
    dbg = 0
    def detect(self):
        return self.objectId
    def action(self):
        return

class trafficSign(scanner_object):
    'traffic sign'
    objects = {} # dictionary
    dbg = 0
    def __init__(self, dbg=0):
        self.dbg = dbg
    def shape_detect(self, src, img, mask):
        ratio = 2
        # filter out the desired component from the light: (g,b) val > mean
        img[mask] = 0
        img_1 = cv2.GaussianBlur(img,(5,5),0)
        # polygon detection  
        #  find contours in the thresholded image and initialize the shape detector
        im2, cnts, hierarchy = cv2.findContours(img_1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # loop over the contours
        for c in cnts:
            shape = "undefined"
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            # detect the name of the shape using only the contour for tiangle, square/rectangle
            if len(approx) == 3:
                shape = "triangle"
            elif len(approx) == 4:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                (x, y, w, h) = cv2.boundingRect(approx)
                ar = w / float(h)
                # a square will have an aspect ratio that is approximately
                # equal to one, otherwise, the shape is a rectangle
                shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            if ( shape != "undefined" ):
                c = c.astype("float")
                c *= ratio
                c = c.astype("int")
                cv2.drawContours(src, [c], 0, (0, 255, 0), 2)
                # compute the center of the contour, then draw the text
                M = cv2.moments(c)
                if (M["m00"] !=0) and (M["m00"] !=0) :
                    cX = int((M["m10"] / M["m00"]) )
                    cY = int((M["m01"] / M["m00"]) )	
                    cv2.putText(src, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    #print("draw %s(%d,%d)\n", shape, cX, cY)
        # circle detection, limit the radius to skip noise circles
        # global thresholding
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=50,maxRadius=250)
        # ensure at least some circles were found
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.putText(src, "circle", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #print("draw circle(%d,%d)\n", i[0], i[1])               
        return
    def detect(self, frame, dbgFrame):
        self.objects.clear()
        resized = cv2.resize(frame ,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_AREA)
        resized_h, resized_w = resized.shape[:2]
        b, g, r = cv2.split(resized)
        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)

        # Red object
        # 1. filter out the red from the light: (g,b) val > mean
        red_mask = np.logical_and( (r < r_mean), (g > g_mean), ( b > b_mean ) )
        img_r = r.copy()
        self.shape_detect(frame, img_r, red_mask)
        if self.dbg == 1 :
            dbgFrame[:,:,2] = img_r
            #cv2.imshow('dbg_frame', dbgFrame)
        return
    def action(self):
        return
    
    
class scanner:
    'scan the frame to get a list of known object: moving and still object'
    objects = {} # dictionary
    dbg = 0
    def __init__(self, dbg=0):
            self.dbg = dbg
    def addObject(self, name, obj):
        self.objects[name] = obj
        return
    def scanObjects(self, frame, dbgFrame):       
        for key in self.objects.keys():
            obj = self.objects[key]
            obj.detect(frame, dbgFrame)
        return
    
dbg = 1    
cap = camera(dbg)
scan = scanner(dbg)
ts = trafficSign(dbg)
# add desired objects to be scanned
scan.addObject('trafficSign', ts) # traffic sig


while( cap.getFrame() ):
    # Convert
    intensity = cap.lightIntensity(cap.currentFrame)
    cv2.putText( cap.currentFrame, str(intensity), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    scan.scanObjects(cap.currentFrame, cap.dbgFrame)
    cv2.imshow('frame', cap.currentFrame)
    if cap.dbg == 1 :
        cv2.imshow('dbg_frame', cap.dbgFrame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
