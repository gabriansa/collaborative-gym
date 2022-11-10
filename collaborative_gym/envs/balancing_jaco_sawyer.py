import numpy as np
import random
import pybullet as p
from .base_env import BaseEnv
from .agents.objects import Object 
from .agents.tool import Tool
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Import all robots
from .agents.jaco import Jaco
from .agents.sawyer import Sawyer
from .agents.panda import Panda

class BalanceTaskJacoSawyerEnv(BaseEnv, MultiAgentEnv):
    def __init__(self):
        self.my_robots = {}
        self.obs_len_robots = {}
        self.gripper_enabled_robots = {}

        # NOTE: Choose the number and type of robots to use in the simulation
        self.my_robots['robot_1'] = Jaco()
        self.my_robots['robot_2'] = Sawyer()

        # NOTE: Define observation lengths for each robot
        self.obs_len_robots['robot_1'] = 34
        self.obs_len_robots['robot_2'] = 34

        # NOTE: Enable or disable gripping for each robot
        self.gripper_enabled_robots['robot_1'] = False
        self.gripper_enabled_robots['robot_2'] = False

        super(BalanceTaskJacoSawyerEnv, self).__init__()

    def step(self, action):
        self.take_step(action)
        # Get observations
        all_observations = self._get_obs()

        # Get rewards
        all_rewards, all_info = self.compute_rewards(action)

        # Get dones
        all_dones = {}
        for robot_name, robot in self.my_robots.items():
            all_dones[robot_name] = self.iteration >= 200
        all_dones['__all__'] = self.iteration >= 200
        
        return all_observations, all_rewards, all_dones, all_info
    
    def compute_rewards(self, action):
        all_rewards = {}
        info = {}

        # Useful variables
        sphere_pos, sphere_orient = self.sphere.get_base_pos_orient()
        balancing_board_center_pos, balancing_board_center_orient = self.balancing_board.get_base_pos_orient()

        # Reward robot 1 and robot 2
        dist_sphere_to_balancing_board_center = -np.linalg.norm(balancing_board_center_pos - sphere_pos)

        # Penalty if ball falls
        if (sphere_pos[2] - balancing_board_center_pos[2]) < -0.009:
            penatly_ball_fell = -1
        else:
            penatly_ball_fell = 0

        all_rewards['robot_1'] = dist_sphere_to_balancing_board_center + penatly_ball_fell
        all_rewards['robot_2'] = dist_sphere_to_balancing_board_center + penatly_ball_fell

        # Get all info
        info['robot_1'] = {"ball_fell": penatly_ball_fell, "dist_ball_to_target": dist_sphere_to_balancing_board_center}
        info['robot_2'] = {"ball_fell": penatly_ball_fell, "dist_ball_to_target": dist_sphere_to_balancing_board_center}
        all_info = self.get_all_info(info)
        
        return all_rewards, all_info

    def _get_obs(self, agent=None):
        # NOTE: Make sure the observation lenghts reflect what is defined at the top --> self.obs_len_robots[<robot_name>]
        all_observations = {}

        # Useful variables
        joint_angles_rob_1 = self.my_robots['robot_1'].get_joint_angles(self.my_robots['robot_1'].controllable_joint_indices)
        joint_angles_rob_2 = self.my_robots['robot_2'].get_joint_angles(self.my_robots['robot_2'].controllable_joint_indices)
        sphere_pos, sphere_orient = self.sphere.get_base_pos_orient()
        balancing_board_center_pos, balancing_board_center_orient = self.balancing_board.get_base_pos_orient()
        sphere_linear_velocity, sphere_angular_velocity = p.getBaseVelocity(self.sphere.body, physicsClientId=self.id)
        
        # Robot 1 observations
        obs_robot_1 = np.concatenate([joint_angles_rob_1, joint_angles_rob_2, sphere_pos, sphere_orient, sphere_linear_velocity, sphere_angular_velocity, balancing_board_center_pos, balancing_board_center_orient]).ravel()
        all_observations['robot_1'] = obs_robot_1

        # Robot 2 observations
        obs_robot_2 = np.concatenate([joint_angles_rob_1, joint_angles_rob_2, sphere_pos, sphere_orient, sphere_linear_velocity, sphere_angular_velocity, balancing_board_center_pos, balancing_board_center_orient]).ravel()
        all_observations['robot_2'] = obs_robot_2

        if agent is not None:
            return all_observations[agent]

        return all_observations

    def reset(self):
        super(BalanceTaskJacoSawyerEnv, self).reset()
        self.create_world()

        # Position robot 1 on a table
        self.my_robots['robot_1'].set_base_pos_orient([0,-1.4,0.65], [0,0,np.pi/2])
        self.table = Object()
        self.table.init('table', self.directory, self.id, self.np_random)
        self.table.set_base_pos_orient([0,-1.65,0], [0,0,np.pi/2])

        # Position robot 2
        self.my_robots['robot_2'].set_base_pos_orient([0,1.8,1], [0,0,-np.pi/2])

        # Create and position a balancing board
        self.create_balancing_board()
        
        # Create and randomly position a sphere on the board
        self.sphere = Object()
        self.sphere.init('sphere', self.directory, self.id, self.np_random, enable_gripping=False)
        balancing_board_center_pos, balancing_board_center_orient = self.balancing_board.get_base_pos_orient()
        x_init_pos_sphere = balancing_board_center_pos[0] + random.uniform(-0.2, 0.2)
        y_init_pos_sphere = balancing_board_center_pos[1] + random.uniform(-0.45, 0.45)
        z_init_pos_sphere = balancing_board_center_pos[2] + 0.05
        self.sphere.set_base_pos_orient([x_init_pos_sphere,y_init_pos_sphere,z_init_pos_sphere], [0,0,np.pi])

        p.resetDebugVisualizerCamera(cameraDistance=2.85, cameraYaw=90, cameraPitch=-10, cameraTargetPosition=[0, 0, 1], physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        #Initialize variables
        self.init_env_variables()
        return self._get_obs()

    def create_balancing_board(self):
        self.balancing_board = Object()
        self.balancing_board.init('balancing_board', self.directory, self.id, self.np_random, enable_gripping=False)
        self.balancing_board.set_base_pos_orient([0,0,0.6], [0,0,0])

        self.disable_collision(obj_1=self.my_robots['robot_1'], obj_2=self.balancing_board)
        self.disable_collision(obj_1=self.my_robots['robot_2'], obj_2=self.balancing_board)

        handle_1_pos, handle_1_orient = self.balancing_board.get_pos_orient(1)
        self.set_end_effector_pos(self.my_robots['robot_1'], handle_1_pos, handle_1_orient)
        p.createConstraint(self.my_robots['robot_1'].body, self.my_robots['robot_1'].end_effector, self.balancing_board.body, 1, p.JOINT_FIXED, [0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, -0.05], parentFrameOrientation=[0,0,0,1], childFrameOrientation=[0, 0, 0, 1], physicsClientId=self.id)

        handle_2_pos, handle_2_orient = self.balancing_board.get_pos_orient(0)
        self.set_end_effector_pos(self.my_robots['robot_2'], handle_2_pos, handle_2_orient)
        p.createConstraint(self.my_robots['robot_2'].body, self.my_robots['robot_2'].end_effector, self.balancing_board.body, 0, p.JOINT_FIXED, [0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, -0.05], parentFrameOrientation=[0,0,0,1], childFrameOrientation=[0, 0, 0, 1], physicsClientId=self.id)
         
    def get_all_info(self, info):
        self.reward_threshold = 0.09

        if abs(info['robot_1']["dist_ball_to_target"]) < self.reward_threshold and self.iteration>5:
            self.task_success_switch = True 

        for robot_name, robot in self.my_robots.items():
            # Check if sucessful task compeltion
            if info[robot_name]["ball_fell"]==0:
                self.task_success[robot_name] += 1
            info[robot_name]['task_success'] = self.task_success[robot_name]
            info[robot_name]['task_performance_%'] = self.task_success[robot_name]/200
            # info[robot_name]['task_completion'] = 1 if self.task_success[robot_name] > 0 else 0
            info[robot_name]['task_completion'] = int(self.task_success_switch)
        return info