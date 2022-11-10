import os
import pybullet as p
import numpy as np
from .agent import Agent

class Object(Agent):
    instances = []
    def __init__(self):
        self.__class__.instances.append(self)
        super(Object, self).__init__()

    def init(self, object_type, directory, id, np_random, enable_gripping=False):
        self.enable_gripping = enable_gripping
        if object_type == 'table':
            object = p.loadURDF(os.path.join(directory, 'table', 'table.urdf'), basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif object_type == 'table_long':
            object = p.loadURDF(os.path.join(directory, 'table_long', 'table_long.urdf'), basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif object_type == 'cube':
            object = p.loadURDF(os.path.join(directory, 'cube', 'cube.urdf'), basePosition=[0.25, -1.0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif object_type == 'big_cube':
            object = p.loadURDF(os.path.join(directory, 'big_cube', 'big_cube.urdf'), basePosition=[0.25, -1.0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif object_type == 'pot':
            object = p.loadURDF(os.path.join(directory, 'pot', 'pot.urdf'), basePosition=[0.25, -1.0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif object_type == 'balancing_board':
            object = p.loadURDF(os.path.join(directory, 'balancing_board', 'balancing_board.urdf'), basePosition=[0.25, -1.0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif object_type == 'sphere':
            object = p.loadURDF(os.path.join(directory, 'sphere', 'sphere.urdf'), basePosition=[0.25, -1.0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        elif object_type == 'big_sphere':
            object = p.loadURDF(os.path.join(directory, 'big_sphere', 'big_sphere.urdf'), basePosition=[0.25, -1.0, 0], baseOrientation=[0, 0, 0, 1], physicsClientId=id)
        else:
            pass

        super(Object, self).init(object, id, np_random, indices=-1)