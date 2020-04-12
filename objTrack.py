import cv2

font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier('banana_clasiffier.xml')
cap = cv2.VideoCapture('input/video4.mp4')
count = 0
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 20)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, 'Banana', (x, y), font, 0.5, (11, 255, 255),
                    2, cv2.LINE_AA)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('img', img)
    if cv2.waitKey(30) & 0xFF == ord('s'):
        cv2.imwrite('output/objectDetect{}.png'.format(count), img)
        count += 1

    # For creating the video
    # cv2.imwrite('output/videoImDet/objectDetect{}.png'.format(count), img)
    # count += 1

cap.release()
cv2.destroyAllWindows()
