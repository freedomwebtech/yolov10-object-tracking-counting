import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import cvzone
from tracker import Tracker
# Load YOLOv8 segmentation model
model = YOLO("yolov9c.pt")  # You might need to adjust the model path

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('vid.mp4')
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count=0
tracker=Tracker()
cy1=479
cy2=510
offset=8
down={}
listcardown=[]
up={}
listcarup=[]
while True:
    ret,frame = cap.read()
    if not ret:
       break
    count += 1
    if count % 5 != 0:
        continue
    
    # Resize frame
    frame = cv2.resize(frame, (1020, 600))
    
    # Perform segmentation
    results = model(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    list=[]
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
    bbox_idx=tracker.update(list)
    for bbox in bbox_idx:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if cy1<(cy+offset) and cy1>(cy-offset):
           down[id]=(cx,cy)
        if id in down:
           if cy2<(cy+offset) and cy2>(cy-offset): 
              cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
              cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
              cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
              if listcardown.count(id)==0:
                 listcardown.append(id)
##########################################CARDOWN#################################
        if cy2<(cy+offset) and cy2>(cy-offset):
           up[id]=(cx,cy)
        if id in up:
           if cy1<(cy+offset) and cy1>(cy-offset):
              cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
              cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)
              cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
              if listcarup.count(id)==0:
                 listcarup.append(id)
            
    cv2.line(frame,(370,479),(684,479),(255,255,255),1)
    cv2.line(frame,(362,510),(684,510),(255,0,0),1)
    cardown=len(listcardown)
    carup=len(listcarup)
    cvzone.putTextRect(frame,f'Cardown:-{cardown}',(50,60),2,2)
    cvzone.putTextRect(frame,f'Carup:-{carup}',(50,160),2,2)

    cv2.imshow("RGB", frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

