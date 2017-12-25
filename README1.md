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
[image3]: ./calibration_images/example_rock1.jpg 

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!
del!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
Here is an example of how to include an image in your writeup.

The test data provided.
![alt text][image1]

The data have recorded.
![alt text][image11]

Indentify the obstacle which appear dark in the perspective transform  images from camera by using RGB color threshold which
value greater than 160.
Add new function `find_rocks()` return the rock pixels by using RGB color threshold which value greater than the red and green
,less than the blue. 

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
And another! 

In the function `perspect_transform()`: Using the cv function `warpPerspective` create a mask to show the field of view of
camera. 

In the function `perspect_transform()`: 

Jupyter Notebook
Rover_Project_Test_Notebook (autosaved) [Python [default]] 

Python [default]

    File
    Edit
    View
    Insert
    Cell
    Kernel
    Widgets
    Help

Rover Project Test Notebook

This notebook contains the functions from the lesson and provides the scaffolding you need to test out your mapping methods. The steps you need to complete in this notebook for the project are the following:

    First just run each of the cells in the notebook, examine the code and the results of each.
    Run the simulator in "Training Mode" and record some data. Note: the simulator may crash if you try to record a large (longer than a few minutes) dataset, but you don't need a ton of data, just some example images to work with.
    Change the data directory path (2 cells below) to be the directory where you saved data
    Test out the functions provided on your data
    Write new functions (or modify existing ones) to report and map out detections of obstacles and rock samples (yellow rocks)
    Populate the process_image() function with the appropriate steps/functions to go from a raw image to a worldmap.
    Run the cell that calls process_image() using moviepy functions to create video output
    Once you have mapping working, move on to modifying perception.py and decision.py to allow your rover to navigate and map in autonomous mode!

Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".

Run the next cell to get code highlighting in the markdown cells.

%%HTML

<style> code {background-color : orange !important;} </style>

%matplotlib inline

#%matplotlib qt # Choose %matplotlib qt to plot to an interactive window (note it may show up behind your browser)

# Make some of the relevant imports

import cv2 # OpenCV for perspective transform

import numpy as np

import matplotlib.image as mpimg

import matplotlib.pyplot as plt

import scipy.misc # For saving images as needed

import glob  # For reading in a list of images from a folder

import imageio

imageio.plugins.ffmpeg.download()

​

Quick Look at the Data

There's some example data provided in the test_dataset folder. This basic dataset is enough to get you up and running but if you want to hone your methods more carefully you should record some data of your own to sample various scenarios in the simulator.

Next, read in and display a random image from the test_dataset folder

path = '../test_dataset/IMG/*'

img_list = glob.glob(path)

# Grab a random image and display it

idx = np.random.randint(0, len(img_list)-1)

image = mpimg.imread(img_list[idx])

plt.imshow(image)

<matplotlib.image.AxesImage at 0x7ff7d88b4b70>

Calibration Data

Read in and display example grid and rock sample calibration images. You'll use the grid for perspective transform and the rock image for creating a new color selection that identifies these samples of interest.

# In the simulator you can toggle on a grid on the ground for calibration

# You can also toggle on the rock samples with the 0 (zero) key.  

# Here's an example of the grid and one of the rocks

example_grid = '../calibration_images/example_grid1.jpg'

example_rock = '../calibration_images/example_rock1.jpg'

grid_img = mpimg.imread(example_grid)

rock_img = mpimg.imread(example_rock)

​

fig = plt.figure(figsize=(12,3))

plt.subplot(121)

plt.imshow(grid_img)

plt.subplot(122)

plt.imshow(rock_img)

<matplotlib.image.AxesImage at 0x7ff7d87cf438>

Perspective Transform

Define the perspective transform function from the lesson and test it on an image.

# Define a function to perform a perspective transform

# I've used the example grid image above to choose source points for the

# grid cell in front of the rover (each grid cell is 1 square meter in the sim)

# Define a function to perform a perspective transform

def perspect_transform(img, src, dst):

           

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    mask =  cv2.warpPerspective(np.ones_like(img[:,:,0]), M , (img.shape[1],img.shape[0]))

    return warped, mask

​

​

# Define calibration box in source (actual) and destination (desired) coordinates

# These source and destination points are defined to warp the image

# to a grid where each 10x10 pixel square represents 1 square meter

# The destination box will be 2*dst_size on each side

dst_size = 5 

# Set a bottom offset to account for the fact that the bottom of the image 

# is not the position of the rover but a bit in front of it

# this is just a rough guess, feel free to change it!

bottom_offset = 6

source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])

destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],

                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],

                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 

                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],

                  ])

warped,mask = perspect_transform(grid_img, source, destination)

pig = plt.figure(figsize=(12,3))

plt.subplot(121)

plt.imshow(warped)

plt.subplot(122)

plt.imshow(mask, cmap='gray')

#scipy.misc.imsave('../output/warped_example.jpg', warped)

<matplotlib.image.AxesImage at 0x7ff7d6f61ba8>

Color Thresholding

Define the color thresholding function from the lesson and apply it to the warped image

TODO: Ultimately, you want your map to not just include navigable terrain but also obstacles and the positions of the rock samples you're searching for. Modify this function or write a new function that returns the pixel locations of obstacles (areas below the threshold) and rock samples (yellow rocks in calibration images), such that you can map these areas into world coordinates as well.
Hints and Suggestion:

    For obstacles you can just invert your color selection that you used to detect ground pixels, i.e., if you've decided that everything above the threshold is navigable terrain, then everthing below the threshold must be an obstacle!

    For rocks, think about imposing a lower and upper boundary in your color selection to be more specific about choosing colors. You can investigate the colors of the rocks (the RGB pixel values) in an interactive matplotlib window to get a feel for the appropriate threshold range (keep in mind you may want different ranges for each of R, G and B!). Feel free to get creative and even bring in functions from other libraries. Here's an example of color selection using OpenCV.

    Beware However: if you start manipulating images with OpenCV, keep in mind that it defaults to BGR instead of RGB color space when reading/writing images, so things can get confusing.

# Identify pixels above the threshold

# Threshold of RGB > 160 does a nice job of identifying ground pixels only

def color_thresh(img, rgb_thresh=(190, 190, 190)):

    # Create an array of zeros same xy size as img, but single channel

    color_select = np.zeros_like(img[:,:,0])

    # Require that each pixel be above all three threshold values in RGB

    # above_thresh will now contain a boolean array with "True"

    # where threshold was met

    above_thresh = (img[:,:,0] > rgb_thresh[0]) \

                & (img[:,:,1] > rgb_thresh[1]) \

                & (img[:,:,2] > rgb_thresh[2])

    # Index the array of zeros with the boolean array and set to 1

    color_select[above_thresh] = 1

    # Return the binary image

    return color_select

​

threshed = color_thresh(warped)

plt.imshow(threshed, cmap='gray')

#scipy.misc.imsave('../output/warped_threshed.jpg', threshed*255)

<matplotlib.image.AxesImage at 0x7ff7d6f34b00>

Coordinate Transformations

Define the functions used to do coordinate transforms and apply them to an image.

# Define a function to convert from image coords to rover coords

def rover_coords(binary_img):

    # Identify nonzero pixels

    ypos, xpos = binary_img.nonzero()

    # Calculate pixel positions with reference to the rover position being at the 

    # center bottom of the image.  

    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)

    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)

    return x_pixel, y_pixel

​

# Define a function to convert to radial coords in rover space

def to_polar_coords(x_pixel, y_pixel):

    # Convert (x_pixel, y_pixel) to (distance, angle) 

    # in polar coordinates in rover space

    # Calculate distance to each pixel

    dist = np.sqrt(x_pixel**2 + y_pixel**2)

    # Calculate angle away from vertical for each pixel

    angles = np.arctan2(y_pixel, x_pixel)

    return dist, angles

​

# Define a function to map rover space pixels to world space

def rotate_pix(xpix, ypix, yaw):

    # Convert yaw to radians

    yaw_rad = yaw * np.pi / 180

    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

                            

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))

    # Return the result  

    return xpix_rotated, ypix_rotated

​

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 

    # Apply a scaling and a translation

    xpix_translated = (xpix_rot / scale) + xpos

    ypix_translated = (ypix_rot / scale) + ypos

    # Return the result  

    return xpix_translated, ypix_translated

​

​

# Define a function to apply rotation and translation (and clipping)

# Once you define the two functions above this function should work

def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):

    # Apply rotation

    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)

    # Apply translation

    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)

    # Perform rotation, translation and clipping all at once

    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)

    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)

    # Return the result

    return x_pix_world, y_pix_world

​

# Grab another random image

idx = np.random.randint(0, len(img_list)-1)

image = mpimg.imread(img_list[idx])

warped,mask = perspect_transform(image, source, destination)

threshed = color_thresh(warped)

​

# Calculate pixel values in rover-centric coords and distance/angle to all pixels

xpix, ypix = rover_coords(threshed)

dist, angles = to_polar_coords(xpix, ypix)

mean_dir = np.mean(angles)

​

# Do some plotting

fig = plt.figure(figsize=(12,9))

plt.subplot(221)

plt.imshow(image)

plt.subplot(222)

plt.imshow(warped)

plt.subplot(223)

plt.imshow(threshed, cmap='gray')

plt.subplot(224)

plt.plot(xpix, ypix, '.')

plt.ylim(-160, 160)

plt.xlim(0, 160)

arrow_length = 100

x_arrow = arrow_length * np.cos(mean_dir)

y_arrow = arrow_length * np.sin(mean_dir)

plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)

​

​

<matplotlib.patches.FancyArrow at 0x7ff7d6e74f28>

def find_rocks(img, levels=(110, 110, 50)) :

    rockpix = ((img[:,:,0] > levels[0]) & (img[:,:,1] > levels[1]) & (img[:,:,2] < levels[2]))

            

    color_select= np.zeros_like(img[:,:,0])

    color_select[rockpix] = 1

    

    return color_select

​

rock_map = find_rocks(rock_img)

fig = plt.figure(figsize=(12,3))

plt.subplot(121)

plt.imshow(rock_img)

plt.subplot(122)

plt.imshow(rock_map, cmap='gray')

            

<matplotlib.image.AxesImage at 0x7ff7d6dc9a90>

Read in saved data and ground truth map of the world

The next cell is all setup to read your saved data into a pandas dataframe. Here you'll also read in a "ground truth" map of the world, where white pixels (pixel value = 1) represent navigable terrain.

After that, we'll define a class to store telemetry data and pathnames to images. When you instantiate this class (data = Databucket()) you'll have a global variable called data that you can refer to for telemetry and map data within the process_image() function in the following cell.

# Import pandas and read in csv file as a dataframe

import pandas as pd

# Change the path below to your data directory

# If you are in a locale (e.g., Europe) that uses ',' as the decimal separator

# change the '.' to ','

df = pd.read_csv('../test_dataset/robot_log.csv', delimiter=';', decimal='.')

csv_img_list = df["Path"].tolist() # Create list of image pathnames

# Read in ground truth map and create a 3-channel image with it

ground_truth = mpimg.imread('../calibration_images/map_bw.png')

ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float)

​

# Creating a class to be the data container

# Will read in saved data from csv file and populate this object

# Worldmap is instantiated as 200 x 200 grids corresponding 

# to a 200m x 200m space (same size as the ground truth map: 200 x 200 pixels)

# This encompasses the full range of output position values in x and y from the sim

class Databucket():

    def __init__(self):

        self.images = csv_img_list  

        self.xpos = df["X_Position"].values

        self.ypos = df["Y_Position"].values

        self.yaw = df["Yaw"].values

        self.count = 0 # This will be a running index

        self.worldmap = np.zeros((200, 200, 3)).astype(np.float)

        self.ground_truth = ground_truth_3d # Ground truth worldmap

​

# Instantiate a Databucket().. this will be a global variable/object

# that you can refer to in the process_image() function below

data = Databucket()

​

Write a function to process stored images

Modify the process_image() function below by adding in the perception step processes (functions defined above) to perform image analysis and mapping. The following cell is all set up to use this process_image() function in conjunction with the moviepy video processing package to create a video from the images you saved taking data in the simulator.

In short, you will be passing individual images into process_image() and building up an image called output_image that will be stored as one frame of video. You can make a mosaic of the various steps of your analysis process and add text as you like (example provided below).

To start with, you can simply run the next three cells to see what happens, but then go ahead and modify them such that the output video demonstrates your mapping process. Feel free to get creative!

The result video :  ./output/Mytest_mapping.mp4

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



![alt text][image3]



