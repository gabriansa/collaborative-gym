import pybullet as p
from .agent import Agent

class Robot(Agent):
    def __init__(self, arm_joint_indices, end_effector, gripper_indices, gripper_collision_indices, tool_joint, tool_pos_offset, tool_orient_offset, opened_gripper, closed_gripper, action_multiplier=1):
        self.arm_joint_indices = arm_joint_indices # Controllable arm joints
        self.controllable_joint_indices = self.arm_joint_indices
        self.end_effector = end_effector # Used to get the pose of the end effector
        self.gripper_indices = gripper_indices # Gripper actuated joints
        self.gripper_collision_indices = gripper_collision_indices
        self.tool_joint = tool_joint
        self.tool_pos_offset = tool_pos_offset
        self.tool_orient_offset = tool_orient_offset
        self.action_multiplier = action_multiplier
        self.opened_gripper = opened_gripper
        self.closed_gripper = closed_gripper
        self.motor_forces = 1.0
        self.motor_gains = 0.05
        super(Robot, self).__init__()


    def init(self, body, id, np_random):
        super(Robot, self).init(body, id, np_random)
        self.arm_lower_limits = [self.lower_limits[i] for i in self.arm_joint_indices]
        self.arm_upper_limits = [self.upper_limits[i] for i in self.arm_joint_indices]
        self.joint_max_forces = self.get_joint_max_force(self.controllable_joint_indices)
        # Determine ik indices for the arm (indices differ since fixed joints are not counted)
        self.arm_ik_indices = []
        for i in self.arm_joint_indices:
            counter = 0
            for j in self.all_joint_indices:
                if i == j:
                    self.arm_ik_indices.append(counter)
                joint_type = p.getJointInfo(self.body, j, physicsClientId=self.id)[2]
                if joint_type != p.JOINT_FIXED:
                    counter += 1