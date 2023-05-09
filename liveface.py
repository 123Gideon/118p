import cv2
image=cv2.VideoCapture(0)
face_classifier=cv2.CascadeClassifier("118.py/haarcascade_frontalface_default.xml")
eyeclassifier=cv2.CascadeClassifier("118.py/haarcascade_eye.xml")
while True:
    dummy,frame=image.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    myfaces=face_classifier.detectMultiScale(gray,1.1,5)
    myeyes=eyeclassifier.detectMultiScale(gray,1.1,5)
    # print(myfaces)
    # print(len(myfaces))




    for x,y,w,h in myfaces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
       
        

    for x,y,w,h in myeyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
       

    cv2.imshow("gid",frame)

    if cv2.waitKey(25)==32:
        break

image.release()
cv2.destroyAllWindows()
