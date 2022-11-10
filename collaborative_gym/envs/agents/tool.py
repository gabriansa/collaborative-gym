import os
import pybullet as p
import numpy as np
from .agent import Agent

class Tool(Agent):
    def __init__(self):
        super(Tool, self).__init__()

    def init(self, robot, tool_name, directory, id, np_random, mesh_scale=[1]*3, maximal=False, alpha=1.0, mass=1):
        self.robot = robot
        self.tool_name = tool_name
        self.id = id
        transforms = self.get_transform()

        # Instantiate the tool mesh
        if tool_name == 'stick':
            tool = p.loadURDF(os.path.join(directory, 'stick', 'stick.urdf'), basePosition=transforms[0], baseOrientation=transforms[1], physicsClientId=id)
        elif tool_name == 'donut':
            tool = p.loadURDF(os.path.join(directory, 'donut', 'donut.urdf'), basePosition=transforms[0], baseOrientation=transforms[1], physicsClientId=id)
        elif tool_name == 'donut_small':
            tool = p.loadURDF(os.path.join(directory, 'donut', 'donut_small.urdf'), basePosition=transforms[0], baseOrientation=transforms[1], physicsClientId=id)
        else:
            tool = None

        super(Tool, self).init(tool, id, np_random, indices=-1)          

        if robot is not None:
            # Disable collisions between the tool and robot
            for j in robot.gripper_collision_indices:
                for tj in self.all_joint_indices + [self.base]:
                    p.setCollisionFilterPair(robot.body, self.body, j, tj, False, physicsClientId=id)
            # Create constraint that keeps the tool in the gripper
            constraint = p.createConstraint(robot.body, robot.tool_joint, self.body, -1, p.JOINT_FIXED, [0, 0, 0], parentFramePosition=self.pos_offset, childFramePosition=[0, 0, 0], parentFrameOrientation=self.orient_offset, childFrameOrientation=[0, 0, 0, 1], physicsClientId=id)
            # p.changeConstraint(constraint, maxForce=5000, physicsClientId=id)

        self.set_gravity(0,0,0)

    def get_transform(self):
        self.robot = self.robot
        if self.robot is not None:
            self.pos_offset = self.robot.tool_pos_offset[self.tool_name]
            self.orient_offset = self.get_quaternion(self.robot.tool_orient_offset[self.tool_name])
            gripper_pos, gripper_orient = self.robot.get_pos_orient(self.robot.tool_joint, center_of_mass=True)
            transform_pos, transform_orient = p.multiplyTransforms(positionA=gripper_pos, orientationA=gripper_orient, positionB=self.pos_offset, orientationB=self.orient_offset, physicsClientId=self.id)
        else:
            transform_pos = [0, 0, 0]
            transform_orient = [0, 0, 0, 1]
        return [transform_pos, transform_orient]

    def reset_pos_orient(self):
        transform_pos, transform_orient = self.get_transform()
        self.set_base_pos_orient(transform_pos, transform_orient)