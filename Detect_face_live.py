import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

print("Press q to quit")

while True:
    
    check , frame = video.read()

    gray_img = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame,scaleFactor = 1.05 , minNeighbors = 5)

    for x ,y , w,h in faces:
        frame = cv2.rectangle(frame , (x,y) , (x+w,y+h) , (255,0,0) , 3)

    cv2.imshow("Face Detection Application",frame)

    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
