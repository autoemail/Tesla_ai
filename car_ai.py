import cv2


img_file="car_img.jpg"

train_model="cars.xml"

train_model_pad="haarcascade_fullbody.xml"

#img=cv2.imread(img_file)

# video=cv2.VideoCapture("vid1.mp4")

# video=cv2.VideoCapture("vid2.mp4")

video=cv2.VideoCapture("vid3.mp4")

# video=cv2.VideoCapture("vid4.mp4")

# video=cv2.VideoCapture("vid5.mp4")

car_train=cv2.CascadeClassifier(train_model)
pedastrain_train=cv2.CascadeClassifier(train_model_pad)

while True:
      (onSuccess,frame)=video.read()
      if onSuccess :
         gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
         
      else:
          break

#       cv2.imshow("Car Ai Detection",gray_frame)
#       cv2.waitKey(1)
      car=car_train.detectMultiScale(gray_frame)
      pad=pedastrain_train.detectMultiScale(gray_frame)
      #print(car)
      for (x,y,w,h) in car:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
          
          
      for (x,y,w,h) in pad:
           cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
          
          
      cv2.namedWindow('Car Ai Detection',cv2.WINDOW_NORMAL)
      cv2.resizeWindow("Car Ai Detection", 800, 600)  
      cv2.imshow("Car Ai Detection",frame)
      cv2.waitKey(1)




print("car scan finished")
