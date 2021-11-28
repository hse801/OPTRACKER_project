# OPTRACKER ♿

### _Personal mobility for the elderly through AI object recognition and tracking and motor drive linkage_


#### ***Photographs of prototypes***

<p align="center">
  <img width="30%" src="https://user-images.githubusercontent.com/76834485/143730232-b5bf2d1c-9228-44b2-82cc-51c871d5ced6.jpg"/>
  <img width="30%" src="https://user-images.githubusercontent.com/76834485/143730438-1c76d6cc-87fa-431e-ae18-54aa6e8a1354.jpg"/>
</p>

#### ***Motion video of the prototype***
<p align="center">
  <img width="70%" src="https://user-images.githubusercontent.com/76834485/143730106-6bb5223e-7c77-476f-9d5f-ee3abcb4cedf.gif"/>
  <img width="70%" src="https://user-images.githubusercontent.com/76834485/143730157-0ec468fb-d2e1-48fe-844b-4f2c8d872bbc.gif"/>
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

