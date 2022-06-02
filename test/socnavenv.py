from mmap import MAP_SHARED
import os
import sys
import numpy as np
import random

from gym import utils
from gym import spaces

import cv2

import pytransform3d as pt3d

import numpy as np
np.random.seed(13)


RESOLUTION = 800
MAP_SIZE = 8.0
MARGIN = 0.5
DEBUG = 'debug' in sys.argv
MAX_TICKS = 400
TIMESTEP = 0.1
ROBOT_RADIUS = 0.15
GOAL_RADIUS = 0.2
GOAL_THRESHOLD = ROBOT_RADIUS + GOAL_RADIUS
NUMBER_OF_HUMANS = 5
HUMAN_THRESHOLD = 0.4

REACH_REWARD = 1.0
OUTOFMAP_REWARD = -0.2
MAXTICKS_REWARD = -0.1
NOTHING_REWARD = -0.0001
COLLISION_REWARD = -1.0

def w2px(i):
    return int(100.*(i+(MAP_SIZE/2)))
def w2py(i):
    return int(100.*((MAP_SIZE/2)-i))


# TO DO List
#
#
# Not urgent:
# - Improve how the robot moves (actual differential platform)
#





class SocNavEnv():
    def __init__(self):
        if DEBUG:
            cv2.namedWindow("world", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("world", RESOLUTION, RESOLUTION)

        self.ticks = 0
        self.humans = np.zeros((NUMBER_OF_HUMANS,4))  # x, y, orientation, speed
        self.robot = np.zeros((1,3))   # x, y, orientation
        self.goal = np.zeros((1,2))    # x, y

        self.robot_is_done = True
        self.world_image = np.zeros((800,800,3))

    @property
    def observation_space(self):
        low  = np.array([-MAP_SIZE/2, -MAP_SIZE/2, -MAP_SIZE/2, -MAP_SIZE/2, -np.pi, ] + [-MAP_SIZE/2, -MAP_SIZE/2, -np.pi, 0.0]*NUMBER_OF_HUMANS  )
        high = np.array([+MAP_SIZE/2, +MAP_SIZE/2, +MAP_SIZE/2, +MAP_SIZE/2, +np.pi, ] + [+MAP_SIZE/2, +MAP_SIZE/2, +np.pi, 1.5]*NUMBER_OF_HUMANS  )
        return spaces.box.Box(low, high, low.shape, np.float32)

    @property
    def action_space(self):
        low  = np.array([0,   -np.pi] )
        high = np.array([1.4, +np.pi] )
        return spaces.box.Box(low, high, low.shape, np.float32)

    @property
    def metadata(self):
        return {}

    @property
    def done(self):
        return self.robot_is_done


    def step(self, action):
        self.ticks += 1

        # update robot
        moved = action[0] * TIMESTEP # speed x time
        xp = self.robot[0,0]+np.sin(self.robot[0,2]+np.pi/2)*moved
        yp = self.robot[0,1]+np.cos(self.robot[0,2]+np.pi/2)*moved
        self.robot[0,0] = xp
        self.robot[0,1] = yp
        self.robot[0,2] += action[1]

        # update humans' positions
        for human in range(self.humans.shape[0]):
            moved = self.humans[human,3] * TIMESTEP # speed x time
            xp = self.humans[human,0]+np.sin(self.humans[human,2]+np.pi/2)*moved
            yp = self.humans[human,1]+np.cos(self.humans[human,2]+np.pi/2)*moved
            self.humans[human,0] = xp
            self.humans[human,1] = yp

        # check for the goal's distance
        distance_to_goal = np.linalg.norm(self.robot[0,0:2]-self.goal[0,0:2], ord=2)

        # check for human-robot collisions
        collision_with_a_human = False
        for human in range(self.humans.shape[0]):
            distance_to_human = np.linalg.norm(self.robot[0,0:2]-self.humans[human,0:2], ord=2)
            if distance_to_human < HUMAN_THRESHOLD:
                collision_with_a_human = True
                break

        # calculate the reward and update is_done
        if MAP_SIZE/2 < self.robot[0,0] or self.robot[0,0] < -MAP_SIZE/2 or MAP_SIZE/2 < self.robot[0,1] or self.robot[0,1] < -MAP_SIZE/2:
            self.robot_is_done = True
            print('Robot left the map')
            reward = -1.0
        elif distance_to_goal < GOAL_THRESHOLD:
            self.robot_is_done = True
            print('Robot got to the goal!!!')
            reward = 1.0
        elif collision_with_a_human is True:
            self.robot_is_done = True
            print('Robot collided with a human')
            reward = -1.0
        elif self.ticks > MAX_TICKS:
            self.robot_is_done = True
            print('MAX TICKS')
            reward = -1.0
        else:
            self.robot_is_done = False
            reward = -0.001

        observation = self.get_observation()
        done = self.robot_is_done
        info = {}

        # self.render()
        return observation, reward, done, info

    def draw_oriented_point(self, input_data, colour):
        left = np.array([input_data[0,0]+np.sin(input_data[0,2])*0.15, input_data[0,1]+np.cos(input_data[0,2])*0.15])
        right = np.array([input_data[0,0]+np.sin(input_data[0,2]+np.pi)*0.15, input_data[0,1]+np.cos(input_data[0,2]+np.pi)*0.15])
        front = np.array([input_data[0,0]+np.sin(input_data[0,2]+np.pi/2)*0.1, input_data[0,1]+np.cos(input_data[0,2]+np.pi/2)*0.1])
        centre = np.array([input_data[0,0], input_data[0,1]])

        cv2.line(self.world_image, (w2px(centre[0]), w2py(centre[1])), (w2px( left[0]), w2py( left[1])), colour, 3)
        cv2.line(self.world_image, (w2px(centre[0]), w2py(centre[1])), (w2px(right[0]), w2py(right[1])), colour, 3)
        cv2.line(self.world_image, (w2px(centre[0]), w2py(centre[1])), (w2px(front[0]), w2py(front[1])), colour, 5)


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
        for i in range(NUMBER_OF_HUMANS):
            self.humans[i,0] = random.uniform(-HALF_SIZE, HALF_SIZE)  # x
            self.humans[i,1] = random.uniform(-HALF_SIZE, HALF_SIZE)  # y
            self.humans[i,2] = random.uniform(-np.pi, np.pi)          # orientation
            self.humans[i,3] = random.uniform(0.0, 1.4)               # speed

        self.robot_is_done = False
        self.ticks = 0

        return self.get_observation()

    def get_observation(self):
        return np.concatenate( (self.robot.flatten(), self.goal.flatten(), self.humans.flatten()) )


    def render(self):
        if DEBUG:
            # empty the image
            self.world_image = (np.ones((800,800,3))*255).astype(np.uint8)
            # draws robot
            self.draw_oriented_point(self.robot, (255, 0, 0))
            # draws goal
            cv2.circle(self.world_image, (w2px(self.goal[0,0]), w2py(self.goal[0,1])), int(GOAL_RADIUS*100.), (0, 255, 0), 2)
            # draws humans
            for human in range(NUMBER_OF_HUMANS):
                self.draw_oriented_point(self.humans[human:human+1,:], (0, 0, 255))
            # shows image
            cv2.imshow("world", self.world_image)
            k = cv2.waitKey(1)
            if k%255 == 27:
                sys.exit(0)

    def close(self):
        pass
