## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./output/test_mapping.jpg
[image11]: ./output/Mytest_mapping.jpg
[image2]: ./output/Results1.png
[image3]: ./output/Results2.png
[image4]: ./output/Results3.png

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

The test data provided.

![alt text][image1]

The data have recorded.

![alt text][image11]

Indentify the obstacle which appear dark in the perspective transform  images from camera by using RGB color threshold which
value greater than 160.
Add new function `find_rocks()` return the rock pixels by using RGB color threshold which value greater than the red and green
,less than the blue. 

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

In the function `perspect_transform()`: Using the cv function `warpPerspective` create a mask to show the field of view of
camera. 

In the function `perspect_transform()`: 
1. Call function `perspect_transform()` return warp and mask.
2. Apply color threshold to idenify navigable terrain and obstacles. Call the function `color_thresh()` return the ground pixels.
3. Get the map of navigable terrain pixels by the threshed map minus one times the mask, outside the camera field  will be zero.
   return the obstacle pixels of the map inside the camera view.
4. Convert to rover coordinates
5. Convert rover-centric pixel values to world coordiates : Call the function `pix_to_world'` conver pixel and coordinates to 
   world coordinates.
6. Update Rover Worldmap : Got world map from above step then update blue channel to 255 and also update red channel to 255 
   where found obstacles.To avoid navigable terrain overlapping the with the obstacles, set the red channel to zero if the blue
   channel is greater than zero.
7. Call `find_rocks()` return the rock pixels and convert it to rover and world coordinates, update the worldmap to display 
   the rock.
8. Output the image and video.   

The result video :  ./output/Mytest_mapping.mp4

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

perception_step() in perception.py
1. Call function `perspect_transform()` return warp and mask.
2. Apply color threshold to idenify navigable terrain and obstacles. Call the function `color_thresh()` return the ground pixels.
3. Get the map of navigable terrain pixels by the threshed map minus one times the mask, outside the camera field  will be zero.
   return the obstacle pixels of the map inside the camera view.
4. Convert to rover coordinates
5. Convert rover-centric pixel values to world coordiates : Call the function `pix_to_world'` conver pixel and coordinates to 
   world coordinates.
6. Update Rover Worldmap : Got world map from above step then update blue channel to 255 and also update red channel to 255 
   where found obstacles.To avoid navigable terrain overlapping the with the obstacles, set the red channel to zero if the blue
   channel is greater than zero.
7. Call `find_rocks()` return the rock pixels and convert it to rover and world coordinates, update the worldmap to display 
   the rock.

decision_step() in `decision.py`

Steering Offset 
        Adjusting the offset based on the mean of the angles with adding a value say -12 let the rover turn right a little bit
        when accelerating. When the rover reach the maximum velocity, adjust the offset based on the standard deviation of the  
        angles and multiplied by and random value between 0.6 and 0.8 to avoid the rover stuck in circle. Adding this offset will 
        let the rover to  hug the left wall. 
        
Create "stuck" mode
        When the rover on forward mode not moving  for at least 4 seconds then turn on "stuck" mode.Also on stop mode turn to
        "stuck" mode when the rover can't forward again.
       
       

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

Adjust steering offset appropriate is important to improve the rover map and the fidelity against the ground truth. Only one
pattern cause the rover stuck in somewhere. I plus some random value a little bit let it have the chance walk through all the
map.

The Results : 
Run the simulator in 1024x768 resolution.Have 45% Mapped, 80% Fidelity in 2 mins ,
95%  Mapped, 77% Fidelity in 10 mins and located 4 rocks.  

![alt text][image2]
![alt text][image3]
![alt text][image4]

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

The rover have only one weapon to dectect the environment and make the decision where to go. Sometimes the RGB color from the
environment let the rover have the wrong way to go. We have to find more weapon help the rover.
