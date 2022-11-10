import numpy as np
import random
import pybullet as p
from .base_env import BaseEnv
from .agents.objects import Object 
from .agents.tool import Tool
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import math
import pyquaternion as pyq

# Import all robots
from .agents.jaco import Jaco
from .agents.sawyer import Sawyer
from .agents.panda import Panda

class PokeTaskSawyersEnv(BaseEnv, MultiAgentEnv):
    def __init__(self):
        self.my_robots = {}
        self.obs_len_robots = {}
        self.gripper_enabled_robots = {}

        # NOTE: Choose the number and type of robots to use in the simulation
        self.my_robots['robot_1'] = Sawyer()
        self.my_robots['robot_2'] = Sawyer()

        # NOTE: Define observation length for each or all robots
        self.obs_len_robots['robot_1'] = 21
        self.obs_len_robots['robot_2'] = 21

        # NOTE: Enable gripping for each or all robots
        self.gripper_enabled_robots['robot_1'] = False
        self.gripper_enabled_robots['robot_2'] = False

        super(PokeTaskSawyersEnv, self).__init__()

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

        # Usefull variables
        stick_pos, stick_orient = self.stick.get_pos_orient(0)
        donut_pos, donut_orient = self.donut.get_pos_orient(0)
        q_stick = pyq.Quaternion(stick_orient[3], stick_orient[0], stick_orient[1], stick_orient[2])
        q_donut = pyq.Quaternion(donut_orient[3], donut_orient[0], donut_orient[1], donut_orient[2])
        qd = q_stick.conjugate * q_donut
        # phi   = math.atan2( 2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x**2 + qd.y**2) )
        theta = math.asin (np.clip(2 * (qd.w * qd.y - qd.z * qd.x), a_min=-1, a_max=1))
        # psi   = math.atan2( 2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y**2 + qd.z**2) )

        # Reward robot 1 and robot 2
        dist_stick_to_donut = -np.linalg.norm(stick_pos - donut_pos)
        moving_penalty_robot_1 = - 0.01*np.linalg.norm(action['robot_1'])
        moving_penalty_robot_2 = - 0.01*np.linalg.norm(action['robot_2'])
        orient_stick_donut = - abs(math.cos(theta))

        # Big reward if stick inside donut
        stick_in_donut_dist_threshold = 0.065
        stick_in_donut_orient_threshold = 0.2 #0.25 for bigger donut
        stick_is_inside_reward = 0
        if abs(dist_stick_to_donut) < stick_in_donut_dist_threshold and abs(orient_stick_donut) < stick_in_donut_orient_threshold:
            stick_is_inside_reward = +1

        all_rewards['robot_1'] = dist_stick_to_donut + 0.8*orient_stick_donut + moving_penalty_robot_1 + stick_is_inside_reward
        all_rewards['robot_2'] = dist_stick_to_donut + 0.8*orient_stick_donut + moving_penalty_robot_2 + stick_is_inside_reward

        # Get all info
        info['robot_1'] = {"dist_stick_to_donut": dist_stick_to_donut}
        info['robot_2'] = {"dist_stick_to_donut": dist_stick_to_donut}
        all_info = self.get_all_info(info)
        
        return all_rewards, all_info

    def _get_obs(self, agent=None):
        # NOTE: Make sure the observation lenghts reflect what is defined at the top --> self.obs_len_robots[<robot_name>]
        all_observations = {}

        # Useful variables
        stick_pos, stick_orient = self.stick.get_pos_orient(0)
        donut_pos, donut_orient = self.donut.get_pos_orient(0)
        joint_angles_rob_1 = self.my_robots['robot_1'].get_joint_angles(self.my_robots['robot_1'].controllable_joint_indices)
        joint_angles_rob_2 = self.my_robots['robot_2'].get_joint_angles(self.my_robots['robot_2'].controllable_joint_indices)
        finger_COM_pos_rob_1, finger_COM_orient_rob_1 = self.my_robots['robot_1'].get_finger_COM()
        finger_COM_pos_rob_2, finger_COM_orient_rob_2 = self.my_robots['robot_2'].get_finger_COM()

        # Robot 1 observations
        obs_robot_1 = np.concatenate([joint_angles_rob_1, stick_pos, stick_orient, donut_pos, donut_orient]).ravel()
        all_observations['robot_1'] = obs_robot_1

        # Robot 2 observations
        obs_robot_2 = np.concatenate([joint_angles_rob_2, stick_pos, stick_orient, donut_pos, donut_orient]).ravel()
        all_observations['robot_2'] = obs_robot_2


        if agent is not None:
            return all_observations[agent]

        return all_observations

    def reset(self):
        super(PokeTaskSawyersEnv, self).reset()
        self.create_world()

        # Position robot 1
        self.my_robots['robot_1'].set_base_pos_orient([0,-1,1], [0,0,np.pi/3])

        # Position robot 2
        self.my_robots['robot_2'].set_base_pos_orient([0,1,1], [0,0,-np.pi/3])

        # Randomize initial robot joint angles
        # self.randomize_init_joint_angles(min_dist=0.5)
        self.set_end_effector_pos(self.my_robots['robot_1'], [0,-1,2.5])
        self.set_end_effector_pos(self.my_robots['robot_2'], [0,1,2.5])

        # Place stick and donut tool on robot_1 and robot_2 respectively
        self.stick = Tool()
        self.stick.init(self.my_robots['robot_1'], 'stick', self.directory, self.id, self.np_random)
        self.donut = Tool()
        self.donut.init(self.my_robots['robot_2'], 'donut', self.directory, self.id, self.np_random)
        
        p.resetDebugVisualizerCamera(cameraDistance=2.45, cameraYaw=90, cameraPitch=-10, cameraTargetPosition=[0, 0, 1], physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        #Initialize variables
        self.init_env_variables()
        return self._get_obs()

    def get_all_info(self, info):
        self.reward_threshold = 0.09
        # Check if sucessful task compeltion
        if abs(info['robot_1']["dist_stick_to_donut"]) < self.reward_threshold:
            self.task_success_switch = True
        if self.task_success_switch:
            self.task_success_clock += 1
            for robot_name, robot in self.my_robots.items():
                self.task_success[robot_name] += int(abs(info[robot_name]["dist_stick_to_donut"]) < self.reward_threshold)        
        for robot_name, robot in self.my_robots.items():
            info[robot_name]['task_performance_%'] = self.task_success[robot_name]/self.task_success_clock if self.task_success_switch else 0
            info[robot_name]['task_completion'] = 1 if self.task_success[robot_name] > 0 else 0
        return info