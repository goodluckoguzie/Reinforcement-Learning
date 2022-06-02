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


RESOLUTION = 700.
RESOLUTION_VIEW = 1000.
MAP_SIZE = 8.0
PIXEL_TO_WORLD = RESOLUTION / MAP_SIZE
MARGIN = 0.5
MAX_TICKS = 250
TIMESTEP = 0.1
ROBOT_RADIUS = 0.15
GOAL_RADIUS = 0.5
GOAL_THRESHOLD = ROBOT_RADIUS + GOAL_RADIUS
NUMBER_OF_HUMANS = 5
HUMAN_THRESHOLD = 0.4

REACH_REWARD = 1.0
OUTOFMAP_REWARD = -0.2
MAXTICKS_REWARD = -0.1
NOTHING_REWARD = -0.0001
COLLISION_REWARD = -1.0

MAX_ADVANCE = 1.4
MAX_ROTATION = np.pi/8


DEBUG = 0
if 'debug' in sys.argv or "debug=2" in sys.argv:
    DEBUG = 2
elif "debug=1" in sys.argv:
    DEBUG = 1


# TO DO List
#
# Urgent:
# - Observations shouldn't have angles, but sin/cos.
#
# Not urgent:
# - Improve how the robot moves (actual differential platform)
#





class SocNavEnv():
    def __init__(self):
        self.window_initialised = False

        self.ticks = 0
        self.humans = np.zeros((NUMBER_OF_HUMANS,4))  # x, y, orientation, speed
        self.robot = np.zeros((1,3))   # x, y, orientation
        self.goal = np.zeros((1,2))    # x, y

        self.robot_is_done = True
        self.world_image = np.zeros((int(RESOLUTION),int(RESOLUTION),3))

    @property
    def observation_space(self):
        low  = np.array([-MAP_SIZE/2, -MAP_SIZE/2, -1.0, -1.0,                                   # robot:  x, y, sin, cos
                         -MAP_SIZE/2, -MAP_SIZE/2] +                                             # goal:   x, y
                        [-MAP_SIZE/2, -MAP_SIZE/2, -1.0, -1.0, 0.0]*NUMBER_OF_HUMANS  )          # humans: x, y, sin, cos, speed
        high = np.array([+MAP_SIZE/2, +MAP_SIZE/2, +1.0, +1.0,                                   # robot:  x, y, sin, cos
                         +MAP_SIZE/2, +MAP_SIZE/2] +                                             # goal:   x, y
                        [+MAP_SIZE/2, +MAP_SIZE/2, +1.0, +1.0, MAX_ADVANCE]*NUMBER_OF_HUMANS  )  # humans: x, y, sin, cos, speed
        return spaces.box.Box(low, high, low.shape, np.float32)

    @property
    def action_space(self):
        low  = np.array([          0, -MAX_ROTATION] )
        high = np.array([MAX_ADVANCE, +MAX_ROTATION] )
        return spaces.box.Box(low, high, low.shape, np.float32)

    @property
    def metadata(self):
        return {}

    @property
    def done(self):
        return self.robot_is_done



    def step(self, action_pre):
        def process_action(action_pre):
            action = np.array(action_pre)
            if action[0] < 0:
                action[0] = 0
            elif action[0] > MAX_ADVANCE:
                action[0] = MAX_ADVANCE
            if action[1] < -MAX_ROTATION:
                action[1] = -MAX_ROTATION
            elif action[1] > MAX_ROTATION:
                action[1] = MAX_ROTATION
            return action

        def update_robot_and_humans(self, action):
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


        action = process_action(action_pre)
        update_robot_and_humans(self, action)

        observation = self.get_observation()
        reward = self.compute_reward_and_ticks()
        done = self.robot_is_done
        info = {}

        if DEBUG > 0 and self.ticks%50==0:
            self.render()
        elif DEBUG > 1:
            self.render()

        return observation, reward, done, info


    def compute_reward_and_ticks(self):
        self.ticks += 1

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
            # print('Robot left the map')
            reward = -1.0
        elif distance_to_goal < GOAL_THRESHOLD:
            self.robot_is_done = True
            # print('Robot got to the goal!!!')
            reward = 1.0
        elif collision_with_a_human is True:
            self.robot_is_done = True
            # print('Robot collided with a human')
            reward = -1.0
        elif self.ticks > MAX_TICKS:
            self.robot_is_done = True
            # print('MAX TICKS')
            reward = -1.0
        else:
            self.robot_is_done = False
            reward = -distance_to_goal/1000

        return reward


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
        def observation_with_cos_sin_rather_than_angle(i):
            output = np.empty((i.shape[0], i.shape[1]+1))
            output[:, 0:2] =        i[:, 0:2]
            output[:,   2] = np.sin(i[:,   2])
            output[:,   2] = np.sin(i[:,   2])
            if i.shape[1] > 3:
                output[:, 3] = i[:, 3]
            return output.flatten()

        robot_obs = observation_with_cos_sin_rather_than_angle(self.robot)
        goal_obs = self.goal.flatten()
        humans_obs = observation_with_cos_sin_rather_than_angle(self.humans)
        return np.concatenate( (robot_obs, goal_obs, humans_obs) )


    def render(self):
        def w2px(i):
            return int(PIXEL_TO_WORLD*(i+(MAP_SIZE/2)))

        def w2py(i):
            return int(PIXEL_TO_WORLD*((MAP_SIZE/2)-i))

        def draw_oriented_point(image, input_data, colour, radius=0.15, nose=0.1):
            left =   np.array([input_data[0,0]+np.sin(input_data[0,2])*      radius, input_data[0,1]+np.cos(input_data[0,2])*      radius])
            right =  np.array([input_data[0,0]+np.sin(input_data[0,2]+np.pi)*radius, input_data[0,1]+np.cos(input_data[0,2]+np.pi)*radius])
            front =  np.array([input_data[0,0]+np.sin(input_data[0,2]+np.pi/2)*nose, input_data[0,1]+np.cos(input_data[0,2]+np.pi/2)*nose])
            centre = np.array([input_data[0,0],                                      input_data[0,1]])

            cv2.line(image, (w2px(centre[0]), w2py(centre[1])), (w2px( left[0]), w2py( left[1])), colour, 3)
            cv2.line(image, (w2px(centre[0]), w2py(centre[1])), (w2px(right[0]), w2py(right[1])), colour, 3)
            cv2.line(image, (w2px(centre[0]), w2py(centre[1])), (w2px(front[0]), w2py(front[1])), colour, 5)


        if not self.window_initialised:
            cv2.namedWindow("world", cv2.WINDOW_NORMAL) 
            cv2.resizeWindow("world", int(RESOLUTION_VIEW), int(RESOLUTION_VIEW))
            self.window_initialised = True

        # empty the image
        self.world_image = (np.ones((int(RESOLUTION),int(RESOLUTION),3))*255).astype(np.uint8)
        # draws robot
        draw_oriented_point(self.world_image, self.robot, (255, 0, 0))
        # draws goal
        cv2.circle(self.world_image, (w2px(self.goal[0,0]), w2py(self.goal[0,1])), int(GOAL_RADIUS*100.), (0, 255, 0), 2)
        # draws humans
        for human in range(NUMBER_OF_HUMANS):
            draw_oriented_point(self.world_image, self.humans[human:human+1,:], (0, 0, 255))
        # shows image
        cv2.imshow("world", self.world_image)
        k = cv2.waitKey(1)
        if k%255 == 27:
            sys.exit(0)

    def close(self):
        pass