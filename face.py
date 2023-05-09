import cv2
image=cv2.imread("118.py/4f.jpg")

gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

face_classifier=cv2.CascadeClassifier("118.py/haarcascade_frontalface_default.xml")

myfaces=face_classifier.detectMultiScale(gray)
print(myfaces)
print(len(myfaces))

for x,y,w,h in myfaces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,0),2)
    roi_image=image[y:y+h,x:x+w]
    cv2.imwrite("4people.png",roi_image)

cv2.imshow("gid",image)
cv2.waitKey(0)
