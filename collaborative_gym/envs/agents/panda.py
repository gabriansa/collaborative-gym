import os
import numpy as np
import pybullet as p
from .robot import Robot

class Panda(Robot):
    def __init__(self):
        arm_joint_indices = [0, 1, 2, 3, 4, 5, 6] # Controllable arm joints
        end_effector = 11 # Used to get the pose of the end effector
        gripper_indices = [9, 10] # Gripper actuated joints
        gripper_collision_indices = [7, 8, 9, 10, 11] # Used to disable collision between gripper and tools
        tool_joint = 11 # Joint that tools are attached to
        tool_pos_offset = {'stick': [0,0,0],
                           'donut': [-0.09,0,0.08],
                           'donut_small': [-0.09,0,0.08]}
        tool_orient_offset = {'stick': [0,0,0],
                              'donut': [0,0,np.pi/2],
                              'donut_small': [0,0,np.pi/2]}
        closed_gripper = [-0.2, -0.2]
        opened_gripper = [0.2, 0.2]

        super(Panda, self).__init__(arm_joint_indices, end_effector, gripper_indices, gripper_collision_indices, tool_joint, tool_pos_offset, tool_orient_offset, opened_gripper, closed_gripper)

    def init(self, directory, id, np_random, fixed_base=True):
        self.body = p.loadURDF(os.path.join(directory, 'robots/panda', 'panda.urdf'), useFixedBase=fixed_base, basePosition=[-1, -1, 0.975], flags=p.URDF_USE_SELF_COLLISION, physicsClientId=id)
        super(Panda, self).init(self.body, id, np_random)

        # Remove collisions between the various arm links for stability
        for i in range(3, 24):
            for j in range(3, 24):
                p.setCollisionFilterPair(self.body, self.body, i, j, 0, physicsClientId=id)
        for i in range(0, 3):
            for j in range(0, 9):
                p.setCollisionFilterPair(self.body, self.body, i, j, 0, physicsClientId=id)

