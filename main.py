import cv2
import numpy as np


face_file = 'D:\\programming\\PyCharm Project\\Eye blink detection\\haarcascade_frontalface_default.xml'
eye_file = 'D:\\programming\\PyCharm Project\\Eye blink detection\\haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv2.CascadeClassifier(face_file)
eye_cascade = cv2.CascadeClassifier(eye_file)

front_read = True

img = cv2.VideoCapture(0)
rec,image = img.read()

while(rec):
    rec,image = img.read()
    color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filter = cv2.bilateralFilter(color, 5, 1, 1)
    
    face = face_cascade.detectMultiScale(color, 1.3, 5, minSize= (200, 200))
    if (len(face)> 0):
        for (x, y , w, h) in face:
            image = cv2.rectangle(image, (x, y), (x + h, y + w), (0, 255, 0), 2)
            
            rio_face = color[y:y + h, x:x + w]
            rio_face_clr = image[y:y + h, x:x + w]
            eye = eye_cascade.detectMultiScale(rio_face, 1.3, 5, minSize= (50, 50))
            
            if (len(eye)>= 2):
                if (front_read):
                    cv2.putText(image,
                                  "Eye detected press s to begin",
                                  (70, 70),
                                  cv2.FONT_HERSHEY_PLAIN, 3,
                                  (0, 255, 0), 2)
                else:
                    cv2.putText(image,
                                "Eyes Open!", (70,70),
                                cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 255, 255), 2)
            else:
                if (front_read):
                    cv2.putText(image,
                                "No Eyes Detected.", (70, 70),
                                cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 2)
                else:
                    print("Blink DEtected--------------!")
                    cv2.waitKey(3000)
                    front_read = True
    else:
        cv2.putText(image,
                    "No Face Detected.", (100,100),
                    cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 2)
    cv2.imshow('image', image)
    wait = cv2.waitKey(1)
    if (wait == ord('q')):
        break
    elif (wait == ord('s') and front_read):
        front_read = False
        
img.release()
cv2.destroyAllWindows()
                    
    



























































