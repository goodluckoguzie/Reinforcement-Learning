from mmap import MAP_SHARED
import os
import sys
import numpy as np
import random

from gym import utils

import cv2

import pytransform3d as pt3d

import numpy as np
np.random.seed(13)


RESOLUTION = 800
MAP_SIZE = 8.0
MARGIN = 0.5
DEBUG = 'debug' in sys.argv
MAX_TICKS = 100

def w2px(i):
    return int(100.*(i+(MAP_SIZE/2)))
def w2py(i):
    return int(100.*((MAP_SIZE/2)-i))


## the action will be advance_speed, rotational_speed


#
#  X1. Improve how humans are drawn
#  X2. Improve hor the robot is drawn
#  3. Make humans move linearly
#  4. Make the robot move based on the action
#  5. Implement the reward
#     * -1.000: if the robot collides with any human, gets out of the map, or MAX_TICKS
#     * +1.000: if the robot reaches the goal (based on a distance threshold)
#     * -0.001: otherwise
#


class SocNavEnv():
    def __init__(self):
        if DEBUG:
            cv2.namedWindow("world", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("world", RESOLUTION, RESOLUTION)

        self.ticks = 0
        self.humans = np.zeros((5,4))  # x, y, orientation, speed
        self.robot = np.zeros((1,3))   # x, y, orientation
        self.goal = np.zeros((1,2))    # x, y

        self.robot_is_done = True
        self.world_image = np.zeros((800,800,3))

    @property
    def done(self):
        return self.robot_is_done


    def step(self, action):
        self.ticks += 1
        if self.ticks > MAX_TICKS:
            self.robot_is_done = True
            for human in range(self.humans.shape[0]):
                self.move(self.humans[human:human+1,:], (0, 0, 255))

        observation = []
        reward = 0
        done = self.robot_is_done
        info = {}

        # DRAWING
        if DEBUG:
            # empty the image
            self.world_image = (np.ones((800,800,3))*255).astype(np.uint8)
            # draws robot
            #self.draw_oriented_point(self.robot, (255, 0, 0))
            # draws goal
            #cv2.circle(self.world_image, (w2px(self.goal[0,0]),  w2py(self.goal[0,1])), 15, (0, 255, 0), 2)
            # draws humans
            for human in range(self.humans.shape[0]):
                self.draw_oriented_point(self.humans[human:human+1,:], (0, 0, 255))
                
            # shows image
            cv2.imshow("world", self.world_image)
            k = cv2.waitKey(1)
            if k%255 == 27:
                sys.exit(0)

        return observation, reward, done, info

    def draw_oriented_point(self, input_data, colour):
        #input_data[0,2] = 1.57
        left = np.array([input_data[0,0]+np.sin(input_data[0,2])*0.15, input_data[0,1]+np.cos(input_data[0,2])*0.15])
        right = np.array([input_data[0,0]+np.sin(input_data[0,2]+np.pi)*0.15, input_data[0,1]+np.cos(input_data[0,2]+np.pi)*0.15])
        front = np.array([input_data[0,0]+np.sin(input_data[0,2]+np.pi/2)*0.1, input_data[0,1]+np.cos(input_data[0,2]+np.pi/2)*0.1])
        centre = np.array([input_data[0,0], input_data[0,1]])
      
        cv2.line(self.world_image, (w2px(centre[0]), w2px(centre[1])), (w2px( left[0]), w2px( left[1])), colour, 3)
        cv2.line(self.world_image, (w2px(centre[0]), w2px(centre[1])), (w2px(right[0]), w2px(right[1])), colour, 3)
        cv2.line(self.world_image, (w2px(centre[0]), w2px(centre[1])), (w2px(front[0]), w2px(front[1])), colour, 5)

    

    

    
    def move(self,input_data,colour):
        input_data[0,3] += 0.5
        







    def reset(self):
        HALF_SIZE = MAP_SIZE/2. - MARGIN
        # robot
        self.robot[0,0] = random.uniform(-HALF_SIZE, HALF_SIZE) # x
        self.robot[0,1] = random.uniform(-HALF_SIZE, HALF_SIZE) # y
        self.robot[0,2] = random.uniform(-np.pi, np.pi)         # orientation

        # goal
        self.goal[0,0] = random.uniform(-HALF_SIZE, HALF_SIZE)  # x
        self.goal[0,1] = random.uniform(-HALF_SIZE, HALF_SIZE)  # y

        # humans
        for i in range(5):
            self.humans[i,0] = random.uniform(-HALF_SIZE, HALF_SIZE)  # x
            self.humans[i,1] = random.uniform(-HALF_SIZE, HALF_SIZE)  # y
            self.humans[i,2] = random.uniform(-np.pi, np.pi)          # orientation
            self.humans[i,3] = random.uniform(0.0, 1.4)               # speed

        self.robot_is_done = False
        self.ticks = 0

        return self.get_observation()

    def get_observation(self):
        return np.concatenate( (self.robot.flatten(), self.goal.flatten(), self.humans.flatten()) )