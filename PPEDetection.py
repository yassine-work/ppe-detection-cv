from ultralytics import YOLO
import cv2
import cvzone
import math
# cap=cv2.VideoCapture(1)  #for webcam
# cap.set(3,1920)
# cap.set(4,1080)
cap=cv2.VideoCapture("../videos/ppe-1.mp4")    #for videos
model=YOLO("../models/ppe.pt")
classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']
myColor=(0,0,255)
while True:
    success,img=cap.read()
    results=model(img,stream=True)#stream for optimisation
    for r in results:
        boxes=r.boxes
        for box in boxes:
            #Bounding Box
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w,h=x2-x1,y2-y1
            #cvzone.cornerRect(img,(x1,y1,w,h))
            # Confidence
            conf=math.ceil((box.conf[0]*100))/100
            #class name
            cls=int(box.cls[0])
            currentClass=classNames[cls]
            if conf>0.5:
                if currentClass=='Hardhat' or currentClass=='Mask' or currentClass=='Safety Vest' :
                    myColor=(0,255,0)
                elif  currentClass=='NO-Hardhat' or currentClass=='NO-Mask' or currentClass=='NO-Safety Vest' :
                    myColor=(255,0,0)
                else:
                    myColor=(0,0,255)
                cv2.rectangle(img,(x1,y1),(x2,y2),myColor,3)
                cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1-35)),
                                   1,1,offset=5,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor)


    cv2.imshow("Image",img)
    cv2.waitKey(1)

