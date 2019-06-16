from __future__ import division
# import the necessary packages
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="video source either file path or RTSP://IPaddr")
args = vars(ap.parse_args())


def nothing(x):
    pass
def lineq(x1,y1,x2,y2,ptx,pty):
    mag=((pty-y1)/(y2-y1))-((ptx-x1)/(x2-x1))
    print (mag)
    if mag<0:
        return mag
    else: return mag
#########declaring list to count cars throug the magnitudes of their centroid wrt line)
magn=[]
prevmagn=[0]
prevmagn.append(0)
firsttime=[0,0]
bkmagn=[]
bkprevmagn=[0]
bkprevmagn.append(0)

nbike=0
ncar=0
#############   
def feed(image,x1,y1,x2,y2,magn,prevmagn,ncar,firsttime,bkmagn,bkprevmagn,nbike):
# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
    buff=[]
    bkbuff=[]
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

# loop over the detections
    for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	    confidence = detections[0, 0, i, 2]

	# filter out weak detections by ensuring the `confidence` is
	# greater than the minimum confidence
	    if confidence > 0.5:
		# extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
                    idx = int(detections[0, 0, i, 1])
                    #print (detections[0,0,i,3:7]* np.array([w, h, w, h]))
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    if idx ==7:
                        val=lineq(x1,y1,x2,y2,(startX+endX)/2,(startY+endY)/2)
		# display the prediction
                        label = "{}: {:.2f}%".format(CLASSES[idx], val)
                        cv2.rectangle(image, (startX, startY), (endX, endY),
			    COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        #cv2.putText(image, label, (startX, y),
			 #   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                        magn.append(val)
                        if  firsttime[0] == 0: prevmagn.append(0)
                    if idx ==2 or idx==14:
                        val=lineq(x1,y1,x2,y2,(startX+endX)/2,(startY+endY)/2)
		# display the prediction
                        label = "{}: {:.2f}%".format(CLASSES[idx], val)
                        cv2.rectangle(image, (startX, startY), (endX, endY),
			    COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        #cv2.putText(image, label, (startX, y),
			 #   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                        bkmagn.append(val)
                        if  firsttime[1] == 0: 
                            bkprevmagn.append(0)
                            firsttime[1]=1
    firsttime[0]=1
    same=0
    buff=magn[:]
    print ("prev",prevmagn)
    print("magn", magn)
    for nbr in buff:
        print ("nbr",nbr)
        if nbr<0:
            #same=0
            for prnbr in prevmagn:
                if prnbr<0:
                    same=1
            if same==0 and len(prevmagn)!=0:
                same=1    
                ncar+=1
                break        
        magn.pop(0)
    bksame=0
    bkbuff=bkmagn[:]
    print ("prev",bkprevmagn)
    print("magn", bkmagn)
    for nbr in bkbuff:
        print ("nbr",nbr)
        if nbr<0:
            #same=0
            for prnbr in bkprevmagn:
                if prnbr<0:
                    bksame=1
            if bksame==0 and len(bkprevmagn)!=0:
                same=1    
                nbike+=1
                break        
        bkmagn.pop(0)        
    bkprevmagn=bkbuff[:]   
    prevmagn=buff[:]   
    print ("cars ={}".format(ncar)) 
    return image,ncar,magn,prevmagn,firsttime,bkmagn,bkprevmagn,nbike


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("toloadmodels/MobileNetSSD_deploy.prototxt.txt", "toloadmodels/MobileNetSSD_deploy.caffemodel")

######Trackbar init####
cv2.namedWindow('track')
x=500
y=500

cv2.createTrackbar("X", "track",0,1000,nothing)
cv2.createTrackbar("Y", "track",500,1500,nothing)
############3
stream_name = "rtsp://admin:admin12345@192.168.1." + args["source"]
vcap=cv2.VideoCapture(stream_name)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
vcap.set(cv2.CAP_PROP_FPS, 30)
fps = vcap.get(cv2.CAP_PROP_FPS)
print ("fps ",fps)
out = cv2.VideoWriter('output/countbncdemo2.avi',fourcc, fps-5, (1880,1024))
#cv2.waitKey(0)
while(True):
    
    ret,image=vcap.read()
    rows=image.shape[0]
    cols=image.shape[1]
    #image=cv2.resize(image,(320,480))
    #image=cv2.resize(image,(cols,rows))
    #image=image[150:image.shape[0],0:image.shape[1]]
    x1=x
    y1=image.shape[0]
    x2=x#image.shape[1]
    y2=y
    image,ncar,magn,prevmagn,firsttime,bkmagn,bkprevmagn,nbike = feed(image,x1,y1,x2,y2,magn,prevmagn,ncar,firsttime,bkmagn,bkprevmagn,nbike)
    image=cv2.resize(image,(1880,1024))
    tex="Cars={}".format(ncar)
    cv2.putText(image,tex,(1000,200),cv2.FONT_HERSHEY_TRIPLEX,3,(255,255,255),5)
    tex1="Bikes={}".format(nbike)
    cv2.putText(image,tex1,(1000,350),cv2.FONT_HERSHEY_TRIPLEX,3,(255,255,255),5)
    cv2.line(image,(x1,y1),(x2,y2),(255,0,0),10)
    if cv2.waitKey(1)==ord('c'):
        break
    out.write(image)
    temp_image = cv2.resize(image,(640,480))
    cv2.imshow("output", temp_image)
    x=cv2.getTrackbarPos("X", "track")
    y=cv2.getTrackbarPos("Y", "track")
vcap.release()
out.release()
cv2.destroyAllWindows()

