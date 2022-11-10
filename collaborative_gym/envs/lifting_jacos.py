import numpy as np
import random
import pybullet as p
from .base_env import BaseEnv
from .agents.objects import Object 
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Import all robots
from .agents.jaco import Jaco
from .agents.sawyer import Sawyer
from .agents.panda import Panda

class LiftTaskJacosEnv(BaseEnv, MultiAgentEnv):
    def __init__(self):
        self.my_robots = {}
        self.obs_len_robots = {}
        self.gripper_enabled_robots = {}

        # NOTE: Choose the number and type of robots to use in the simulation
        self.my_robots['robot_1'] = Jaco()
        self.my_robots['robot_2'] = Jaco()

        # NOTE: Define observation lengths for each robot
        self.obs_len_robots['robot_1'] = 54
        self.obs_len_robots['robot_2'] = 54

        # NOTE: Enable/disable gripping for each robot
        self.gripper_enabled_robots['robot_1'] = True
        self.gripper_enabled_robots['robot_2'] = True

        super(LiftTaskJacosEnv, self).__init__()

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
        handle_1_pos, _ = self.pot.get_pos_orient(1)
        handle_2_pos, _ = self.pot.get_pos_orient(0)
        pot_pos, pot_orient = self.pot.get_pos_orient(-1)
        pot_orient_euler = self.pot.get_euler(pot_orient)
        finger_COM_pos_rob_1, _ = self.my_robots['robot_1'].get_finger_COM()
        finger_COM_pos_rob_2, _ = self.my_robots['robot_2'].get_finger_COM()
        target_pos = self.target_pos

        # Reward robot 1 and robot 2
        dist_rob_1_to_handle_1 = -np.linalg.norm(finger_COM_pos_rob_1 - handle_1_pos)
        dist_rob_2_to_handle_2 = -np.linalg.norm(finger_COM_pos_rob_2 - handle_2_pos)
        dist_pot_to_target = -np.linalg.norm(pot_pos - target_pos)
        pot_tilt = - 0.2*(abs(pot_orient_euler[0]) + abs(pot_orient_euler[1]))
        moving_penalty_robot_1 = - 0.01*np.linalg.norm(action['robot_1'][:len(self.my_robots['robot_1'].arm_joint_indices)])
        moving_penalty_robot_2 = - 0.01*np.linalg.norm(action['robot_2'][:len(self.my_robots['robot_2'].arm_joint_indices)])

        # Gripping incentive to facilitate learning
        w = 0.1
        if dist_rob_1_to_handle_1 < self.threshold_picking:
            gripping_incentive_rob_1 = -w*action['robot_1'][-1]
        else:
            gripping_incentive_rob_1 = -w*1
        if dist_rob_2_to_handle_2 < self.threshold_picking:
            gripping_incentive_rob_2 = -w*action['robot_2'][-1]
        else:
            gripping_incentive_rob_2 = -w*1

        all_rewards['robot_1'] = dist_rob_1_to_handle_1 + 2*dist_pot_to_target + moving_penalty_robot_1 + pot_tilt + gripping_incentive_rob_1
        all_rewards['robot_2'] = dist_rob_2_to_handle_2 + 2*dist_pot_to_target + moving_penalty_robot_2 + pot_tilt + gripping_incentive_rob_2

        # Get all info
        info['robot_1'] = {"dist_pot_to_target": dist_pot_to_target}
        info['robot_2'] = {"dist_pot_to_target": dist_pot_to_target}
        all_info = self.get_all_info(info)
        
        return all_rewards, all_info

    def _get_obs(self, agent=None):
        # NOTE: Make sure the observation lenghts reflect what is defined at the top --> self.obs_len_robots[<robot_name>]
        all_observations = {}

        # Useful variables
        joint_angles_rob_1 = self.my_robots['robot_1'].get_joint_angles(self.my_robots['robot_1'].controllable_joint_indices)
        joint_angles_rob_2 = self.my_robots['robot_2'].get_joint_angles(self.my_robots['robot_2'].controllable_joint_indices)
        handle_1_pos, handle_1_orient = self.pot.get_pos_orient(1)
        handle_2_pos, handle_2_orient = self.pot.get_pos_orient(0)
        pot_pos, pot_orient = self.pot.get_pos_orient(-1)
        finger_COM_pos_rob_1, finger_COM_orient_rob_1 = self.my_robots['robot_1'].get_finger_COM()
        finger_COM_pos_rob_2, finger_COM_orient_rob_2 = self.my_robots['robot_2'].get_finger_COM()
        gripper_status_rob_1 = np.array([int(self.my_robots['robot_1'].ready_to_grip)])
        gripper_status_rob_2 = np.array([int(self.my_robots['robot_2'].ready_to_grip)])
        target_pos = self.target_pos

        # Robot 1 observations
        obs_robot_1 = np.concatenate([joint_angles_rob_1, joint_angles_rob_2, gripper_status_rob_1, gripper_status_rob_2, handle_1_pos, handle_1_orient, handle_2_pos, handle_2_orient, finger_COM_pos_rob_1, finger_COM_orient_rob_1, finger_COM_pos_rob_2, finger_COM_orient_rob_2, pot_pos, pot_orient, target_pos]).ravel()
        all_observations['robot_1'] = obs_robot_1

        # Robot 2 observations
        obs_robot_2 = np.concatenate([joint_angles_rob_1, joint_angles_rob_2, gripper_status_rob_1, gripper_status_rob_2, handle_1_pos, handle_1_orient, handle_2_pos, handle_2_orient, finger_COM_pos_rob_1, finger_COM_orient_rob_1, finger_COM_pos_rob_2, finger_COM_orient_rob_2, pot_pos, pot_orient, target_pos]).ravel()
        all_observations['robot_2'] = obs_robot_2

        if agent is not None:
            return all_observations[agent]

        return all_observations

    def reset(self):
        super(LiftTaskJacosEnv, self).reset()
        self.create_world()

        # Position robot 1 on a table
        self.my_robots['robot_1'].set_base_pos_orient([0,-0.9,0.65], [0,0,np.pi/2])
        self.table_1 = Object()
        self.table_1.init('table', self.directory, self.id, self.np_random)
        self.table_1.set_base_pos_orient([0,-1.1,0], [0,0,np.pi/2])
        # self.set_end_effector_pos(self.my_robots['robot_1'], [0,0,0.8])

        # Position robot 2
        self.my_robots['robot_2'].set_base_pos_orient([0,0.9,0.6], [0,0,-np.pi/2])
        self.table_2 = Object()
        self.table_2.init('table', self.directory, self.id, self.np_random)
        self.table_2.set_base_pos_orient([0,1.1,0], [0,0,np.pi/2])

        # Create and posiiton a table
        self.table_3 = Object()
        self.table_3.init('table', self.directory, self.id, self.np_random)
        self.table_3.set_base_pos_orient([0,0,0], [0,0,np.pi/2])
        
        # Create and posiiton a pot
        self.pot = Object()
        self.pot.init('pot', self.directory, self.id, self.np_random, enable_gripping=True)
        self.pot.set_base_pos_orient([0,0,0.7], [0,0,0])

        # Generate target
        target_pos = np.array([0, 0, 1.2])
        self.create_sphere(radius=0.02, mass=0.0, pos=target_pos, collision=False, rgba=[0, 1, 0, 1])
        self.target_pos = target_pos

        # Randomize the initial joint angles for each robot
        # self.randomize_init_joint_angles(min_dist=0.5)

        self.threshold_picking = 0.04

        p.resetDebugVisualizerCamera(cameraDistance=2.45, cameraYaw=90, cameraPitch=-10, cameraTargetPosition=[0, 0, 1], physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        #Initialize variables
        self.init_env_variables()
        return self._get_obs()

    def get_all_info(self, info):
        self.reward_threshold = 0.15
        # Check if sucessful task compeltion
        if abs(info['robot_1']["dist_pot_to_target"]) < self.reward_threshold:
            self.task_success_switch = True
        if self.task_success_switch:
            self.task_success_clock += 1
            for robot_name, robot in self.my_robots.items():
                self.task_success[robot_name] += int(abs(info[robot_name]["dist_pot_to_target"]) < self.reward_threshold)        
        for robot_name, robot in self.my_robots.items():
            info[robot_name]['task_performance_%'] = self.task_success[robot_name]/self.task_success_clock if self.task_success_switch else 0
            info[robot_name]['task_completion'] = 1 if self.task_success[robot_name] > 0 else 0
        return info