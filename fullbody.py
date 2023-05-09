import cv2
image=cv2.VideoCapture("118.py/walking.avi")
fullbody_classifier=cv2.CascadeClassifier("118.py/haarcascade_fullbody.xml")



# print(myfaces)
# print(len(myfaces))



while True:
     dummy,frame=image.read()
     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     mybody=fullbody_classifier.detectMultiScale(gray,1.2,3)
     for x,y,w,h in mybody:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,0),2)
         cv2.imshow("gid",image)


     if cv2.waitKey(1)==32:
          break
 



       
        

   


   

image.release()
cv2.destroyAllWindows()
