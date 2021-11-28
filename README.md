# OPTRACKER ♿

### _Personal mobility for the elderly through AI object recognition and tracking and motor drive linkage_


#### ***Photographs of prototypes***

<p align="center">
  <img width="25%" src="https://user-images.githubusercontent.com/76834485/143730232-b5bf2d1c-9228-44b2-82cc-51c871d5ced6.jpg"/>
  <img width="25%" src="https://user-images.githubusercontent.com/76834485/143730438-1c76d6cc-87fa-431e-ae18-54aa6e8a1354.jpg"/>
</p>

#### ***Motion video of the prototype***
<p align="center">
  <img width="65%" src="https://user-images.githubusercontent.com/76834485/143730106-6bb5223e-7c77-476f-9d5f-ee3abcb4cedf.gif"/>
  <img width="65%" src="https://user-images.githubusercontent.com/76834485/143730157-0ec468fb-d2e1-48fe-844b-4f2c8d872bbc.gif"/>
 </p>

<br/>

## Contents
[Ⅰ.Introduction](#Introduction)

[Ⅱ.Design Objective](#Design-Objective)

[Ⅲ.Sortware Design](#Software-Design)

[Ⅳ.Hardware Design](#Hardware-Design)

[Ⅴ.Conclusion](#Conclusion)

## Introduction

With the advancement of technology, ***autonomous driving-oriented future mobility*** is becoming a reality, not a distant future. Accordingly, social interest in how future mobility technology can be applied to the transportation disadvantaged is also increasing. In order to move forward to a better future away from the reality where everyone cannot enjoy the advantages of future mobility, various forms of personal mobility and new modes of transportation such as PBV (Purpose Built Vehicle) that can be used by everyone are needed!

<br/>

***Wheelchairs*** show their true value when various functions are added to the purpose or user specificity. Users with visual or hearing difficulties, as well as situations where walking or movement is difficult for physical reasons, can use functions such as autonomous driving to move to their destination without the help of a guardian.

## Design Objective

Personal mobility development uses **AI object recognition and tracking technology** to recognize only one of the various objects captured by the camera in real time, and to **link the coordinate value of the object to motor driving** to drive only the object at a certain distance.

- Implementation of autonomous driving technology at limited cost
- Solving the safety problem of autonomous driving by maintaining distance
- Implementation of object recognition technology that is not limited by special devices

## Software Design

The solution OPTRACKER we propose has ***four main functions***
<br/>
***Recognize a specific object and track the recognized object in real time.*** It drives the ***motor based on the coordinates of the recognized object and operates the manufactured hardware.***

### Object Detection & Tracking
We use the [yolov5](https://github.com/ultralytics/yolov5) algorithm to detect objects in real time, and track the detected objects using the [deepsort](https://github.com/nwojke/deep_sort) algorithm. At this time, in order to define the object to be tracked as one person, an algorithm was used to ***track only the object after first registering the object with the camera.***
<br/>
#### ***Detection Tracking Run Screen***
<p align="center">
  <img width="60%" src="https://user-images.githubusercontent.com/76834485/143732792-c1dc8bad-b39e-4c16-a5cf-5dc087534820.gif"/>
  <img width="60%" src="https://user-images.githubusercontent.com/76834485/143732856-c1390ae1-1b17-4f4b-a212-d122b96ec864.gif"/>
 </p>

### Depth Estimation
In order to keep track of the object at a certain distance, the distance to the object was measured. For distance measurement, we used the [monodepth2](https://github.com/nianticlabs/monodepth2) algorithm. In monodepth2, the depth of an image is estimated in real time through a single image. By estimating depth from a single image, we were able to reduce errors in the calibration process and reduce costs. From the estimated depth information, the object's depth information was extracted based on the object's coordinate values obtained from yolo. ***To match the estimated depth information with the actual absolute distance, we measured the distance and depth information at different settings to obtain a correlation between the two.***   
You can check the measurement results in the graph below.
<br/>
<br/>
***Correlation graph***
<p align="center">
  <img width="50%" src="https://user-images.githubusercontent.com/76834485/143732314-6dd066e2-cfe7-4226-a8f3-6b52cfb34e60.png"/>
</p>

#### ***Motion video showing distance maintenance***  


<p align="center">
  <img width="70%" src="https://user-images.githubusercontent.com/76834485/143730157-0ec468fb-d2e1-48fe-844b-4f2c8d872bbc.gif"/>
</p>
 
It runs while maintaining a certain distance and stops when the object stops and becomes narrower than a certain distance.

## Hardware Design
By linking the user's coordinate value with DC motor driving, OPTRACKER only followed the object in the direction the object moves. The direction of the motor was determined by handing over the position of the central pixel value in the object box detected by yolo to Arduino. Therefore, the op tracker can go straight, right, and left and at this time, the motor was driven by connecting a motor driver to Arduino.

In addition, the distance value between OPTRACKER and the user measured by monodepth2 was handed over to arduino to maintain a certain distance from the user. To develop using these algorithms, Jetson Xavier was used, which is an artificial intelligence (AI) board specialized in video computation, including detection of certain objects and tracking of movements.
<br/>
<br/>
***Pass the value to arduino using Pyserial***


      ARD = serial.Serial('/dev/ttyACM0', 9600)
      if id == id_min:
      if center_x < 180:
          c = "2"
          data = c.encode('utf-8')
          ARD.write(data)
      if center_x > 480:
          c = "1"
          data = c.encode('utf-8')
          ARD.write(data)
      if disp_center < 2.0:
          c = "3"
          data = c.encode('utf-8')
          ARD.write(data)
      else:
          c = "0"
          data = c.encode('utf-8')
          ARD.write(data)  
          
          

## Conclusion

There are three main differences between OPTRACKER and existing personal mobility.  

***First,*** OPTRACKER automatically follows you when you register, so it is easy to control the disabled and the elderly.  

***Second,*** OPTRACKER can easily register and recognize users by applying image processing technology. By using images of cameras mounted on the front in YOLO and Deep Sort algorithms, we solved the problems of existing products that users should not only have characteristics that differentiate them from others but also be accurately recognized by sensors.  

***Finally,*** OPTRACKER can quickly recognize and track users.  


OPTRACTER has very high scalability of the technology itself. It can be applied to cultural facilities, amenities, and wheelchairs outdoors as well as other domains such as shopping carts, carriers, and strollers.  



You can implement all of these things through webcam in real time with that code.
(However, if you want to drive a motor, you need to annotate the serial code and use it.)


    python3 track.py--source 0 --yolo_weights Yolov5_DeepSort_Pytorch/yolov5/weights/yolov5s.pt --model_name mono+stereo_640x192 --img 640 --classes 0 --show-vid --save-txt --image_path assets



## Contributors


|박지은|오지영|이다인|한지은|황시은|
|------|---|---|---|---|
|ㅇㅇ|ㅇㅇ|ㅇㅇ|ㅇ|ㅇ|

