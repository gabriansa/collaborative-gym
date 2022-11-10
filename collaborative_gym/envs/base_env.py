import os, time, gc
import numpy as np
import gym
import random
from gym import spaces
from gym.utils import seeding
from screeninfo import get_monitors
import pybullet as p
from .agents.objects import Object 

from .util import Util
from .agents.agent import Agent

class BaseEnv(gym.Env):
    def __init__(self, time_step=0.02, frame_skip=5, render=False, gravity=-9.81, seed=1001):
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.gravity = gravity
        self.id = None
        self.gui = False
        self.gpu = False
        self.view_matrix = None
        self.seed(seed)
        if render:
            self.render()
        else:
            self.id = p.connect(p.DIRECT)
            self.util = Util(self.id, self.np_random)

        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')
        
        # Define action space for each robot
        self.action_space_robot = {}
        for robot_name, robot_class in self.my_robots.items():
            action_robot_len = len(robot_class.controllable_joint_indices)
            # Add gripper action if gripper is enabled
            if len(self.gripper_enabled_robots) == len(self.my_robots) and self.gripper_enabled_robots[robot_name]:
                action_robot_len += 1
            elif len(self.gripper_enabled_robots) != len(self.my_robots):
                print("Gripper enabling mode for robots needs to be defined for every single robot")
                exit()
            self.action_space_robot[robot_name] = spaces.Box(low=np.array([-1.0]*action_robot_len, dtype=np.float32), high=np.array([1.0]*action_robot_len, dtype=np.float32), dtype=np.float32)
        
        # Define observation space for each robot
        self.observation_space_robot = {}
        for robot_name, robot_class in self.my_robots.items():
            if len(self.obs_len_robots) == len(self.my_robots): 
                obs_robot_len = self.obs_len_robots[robot_name]
            else:
                print("Received observation lenghts for robots needs to be defined for every single robot")
                exit()
            self.observation_space_robot[robot_name] = spaces.Box(low=np.array([-1000000000.0]*obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*obs_robot_len, dtype=np.float32), dtype=np.float32)

        self.plane = Agent()

    def step(self, action):
        raise NotImplementedError('Implement observations')

    def _get_obs(self, agent=None):
        raise NotImplementedError('Implement observations')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_seed(self, seed=1000):
        self.np_random.seed(seed)

    def enable_gpu_rendering(self):
        self.gpu = True

    def disconnect(self):
        p.disconnect(self.id)

    def reset(self):
        p.resetSimulation(physicsClientId=self.id)
        if not self.gui:
            # Reconnect the physics engine to forcefully clear memory when running long training scripts
            self.disconnect()
            self.id = p.connect(p.DIRECT)
            self.util = Util(self.id, self.np_random)
        if self.gpu:
            self.util.enable_gpu()
        # Configure camera position
        p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0, 0, 1], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)
        p.setTimeStep(self.time_step, physicsClientId=self.id)
        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.id)
        self.last_sim_time = None
        self.iteration = 0
        self.task_success_clock = 0
        self.task_success_switch = False
        self.task_success = {}
        for robot_name, robot in self.my_robots.items():
            self.task_success[robot_name] = 0
        self.updatable_objects = {}
        Object.instances = []
        self.threshold_picking = 0.02

    def create_world(self):
        # Load the ground plane
        plane = p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)
        self.plane.init(plane, self.id, self.np_random, indices=-1)
        # Randomly set friction of the ground
        # self.plane.set_frictions(self.plane.base, lateral_friction=self.np_random.uniform(0.025, 0.5), spinning_friction=0, rolling_friction=0)
        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)
        # Create robots
        for _, robot in self.my_robots.items():
            robot.init(self.directory, self.id, self.np_random)
            robot.set_gravity(0, 0, 0)
            finger_COM_pos, _ = robot.get_finger_COM()
            robot.finger_COM_sphere = self.create_sphere(radius=0.003, mass=0.0, pos=finger_COM_pos, collision=False, rgba=[0.5, 0.5, 0.5, 0.5])

    def take_step(self, actions, gains=None, forces=None, action_multiplier=0.05, step_sim=True):
        if self.last_sim_time is None:
            self.last_sim_time = time.time()
        self.iteration += 1
        for i, (robot_name, robot) in enumerate(self.my_robots.items()):

            robot_actions = actions[robot_name].copy()
            robot_actions = np.clip(robot_actions, a_min=self.action_space_robot[robot_name].low, a_max=self.action_space_robot[robot_name].high)
            robot_actions *= action_multiplier
            if len(self.gripper_enabled_robots) == len(self.my_robots) and self.gripper_enabled_robots[robot_name]:
                joint_actions = robot_actions[:-1]
                gripper_action = True if robot_actions[-1]<0 else False
            else:
                joint_actions = robot_actions
    
            joint_actions *= robot.action_multiplier

            # Append the new action to the current measured joint angles
            robot_joint_angles = robot.get_joint_angles(robot.controllable_joint_indices)
            # Update the target robot joint angles based on the proposed action and joint limits
            for _ in range(self.frame_skip):
                below_lower_limits = robot_joint_angles + joint_actions < robot.controllable_joint_lower_limits
                above_upper_limits = robot_joint_angles + joint_actions > robot.controllable_joint_upper_limits
                joint_actions[below_lower_limits] = 0
                joint_actions[above_upper_limits] = 0
                robot_joint_angles[below_lower_limits] = robot.controllable_joint_lower_limits[below_lower_limits]
                robot_joint_angles[above_upper_limits] = robot.controllable_joint_upper_limits[above_upper_limits]
                robot_joint_angles += joint_actions
                
                robot.control(robot.controllable_joint_indices, robot_joint_angles, robot.motor_gains, robot.motor_forces)

                if len(self.gripper_enabled_robots) == len(self.my_robots) and self.gripper_enabled_robots[robot_name]:
                    self.update_grippable_objects(gripper_action, robot_name, robot)
                
        if step_sim:
            # Update all agent positions
            for _ in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)
                self.update_targets()
                self.update_objects()
                self.update_robot_finger_COM()
                if self.gui:
                    # Slow down time so that the simulation matches real time
                    self.slow_time()

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    def update_targets(self):
        pass

    def update_objects(self):
        pass

    def update_grippable_objects(self, gripper_action, robot_name, robot):
        all_distances = []
        if self.gripper_enabled_robots[robot_name]:
            for object_name, obj in self.all_grippable_objects.items():
                for joint in range(-1,p.getNumJoints(obj.body, physicsClientId=self.id)):
                    finger_COM_pos, finger_COM_orien = robot.get_finger_COM()
                    obj_pos, _ = obj.get_pos_orient(joint)
                    dist_finger_COM_to_obj = abs(np.linalg.norm(obj_pos-finger_COM_pos))
                    all_distances.append(abs(dist_finger_COM_to_obj))
                    # When distance is lower than threshold then set robot.grippable[object_name]['grippable'] to True
                    robot.grippable[object_name]['grippable']['joint_'+str(joint)] = True if dist_finger_COM_to_obj < self.threshold_picking else False
                    # If robot is ready to grip and the object is grippable then update its position
                    if robot.grippable[object_name]['grippable']['joint_'+str(joint)] and gripper_action:
                        if robot.grippable[object_name]['constraint']['joint_'+str(joint)] is None:
                            robot.grippable[object_name]['constraint']['joint_'+str(joint)] = p.createConstraint(robot.body, robot.end_effector, obj.body, joint, p.JOINT_POINT2POINT, [0, 0, 0], parentFramePosition=[0,0,0], childFramePosition=[0, 0, 0], parentFrameOrientation=[0,0,0,1], childFrameOrientation=[0, 0, 0, 1], physicsClientId=self.id)
                        # robot.control(robot.gripper_indices, robot.closed_gripper, robot.motor_gains, robot.motor_forces)
                    else:
                        robot.its_gripping = False
                        if robot.grippable[object_name]['constraint']['joint_'+str(joint)] is not None:
                            p.removeConstraint(robot.grippable[object_name]['constraint']['joint_'+str(joint)], physicsClientId=self.id)
                            robot.grippable[object_name]['constraint']['joint_'+str(joint)] = None
                        # robot.control(robot.gripper_indices, robot.opened_gripper, robot.motor_gains, robot.motor_forces)
        robot.visual_gripping = True if any(i<0.03 for i in all_distances) else False
        constraints_list = []
        for object_name, obj in self.all_grippable_objects.items():
            for const_id, const in robot.grippable[object_name]['constraint'].items():
                constraints_list.append(const)
        if all(v is None for v in constraints_list):
            robot.its_gripping = False
            robot.control(robot.gripper_indices, robot.opened_gripper, robot.motor_gains, robot.motor_forces)
            robot.buff = 0
        else:
            robot.its_gripping = True
            if robot.buff == 0 and robot.visual_gripping:
                robot.control(robot.gripper_indices, robot.closed_gripper, robot.motor_gains, robot.motor_forces)
            robot.buff =+ 1

    def update_robot_finger_COM(self):
        for robot_name, robot in self.my_robots.items():
            finger_COM_pos, _ = robot.get_finger_COM()
            robot.finger_COM_sphere.set_base_pos_orient(finger_COM_pos, [0, 0, 0, 1])

    def render(self, mode='human'):
        if not self.gui:
            self.gui = True
            if self.id is not None:
                self.disconnect()
            try:
                self.width = get_monitors()[0].width
                self.height = get_monitors()[0].height
            except Exception as e:
                self.width = 1920
                self.height = 1080
            self.id = p.connect(p.GUI, options='--background_color_red=0.81 --background_color_green=0.93 --background_color_blue=0.99 --width=%d --height=%d' % (self.width, self.height))
            self.util = Util(self.id, self.np_random)

    def get_euler(self, quaternion):
        return np.array(p.getEulerFromQuaternion(np.array(quaternion), physicsClientId=self.id))

    def get_quaternion(self, euler):
        return np.array(p.getQuaternionFromEuler(np.array(euler), physicsClientId=self.id))

    def setup_camera(self, camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrix(camera_eye, camera_target, [0, 0, 1], physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)

    def setup_camera_rpy(self, camera_target=[-0.2, 0, 0.75], distance=1.5, rpy=[0, -35, 40], fov=60, camera_width=1920//4, camera_height=1080//4):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(camera_target, distance, rpy[2], rpy[1], rpy[0], 2, physicsClientId=self.id)
        self.projection_matrix = p.computeProjectionMatrixFOV(fov, camera_width / camera_height, 0.01, 100, physicsClientId=self.id)

    def get_camera_image_depth(self, light_pos=[0, -3, 1], shadow=False, ambient=0.8, diffuse=0.3, specular=0.1):
        assert self.view_matrix is not None, 'You must call env.setup_camera() or env.setup_camera_rpy() before getting a camera image'
        w, h, img, depth, _ = p.getCameraImage(self.camera_width, self.camera_height, self.view_matrix, self.projection_matrix, lightDirection=light_pos, shadow=shadow, lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse, lightSpecularCoeff=specular, physicsClientId=self.id)
        img = np.reshape(img, (h, w, 4))
        depth = np.reshape(depth, (h, w))
        return img, depth

    def create_sphere(self, radius=0.01, mass=0.0, pos=[0, 0, 0], visual=True, collision=True, rgba=[0, 1, 1, 1], maximal_coordinates=False, return_collision_visual=False):
        sphere_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius, physicsClientId=self.id) if collision else -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=rgba, physicsClientId=self.id) if visual else -1
        if return_collision_visual:
            return sphere_collision, sphere_visual
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=pos, useMaximalCoordinates=maximal_coordinates, physicsClientId=self.id)
        sphere = Agent()
        sphere.init(body, self.id, self.np_random, indices=-1)
        return sphere

    def randomize_init_joint_angles(self, min_dist=0.5, radius=2, joint_randomness=0.15):
        done = False
        while not done:
            # random_angles = {}
            # # Generate random angles for each robot
            # for robot_name, robot in self.my_robots.items():
            #     random_angles[robot_name] = []
            #     for joint in robot.arm_joint_indices:
            #         random_angles[robot_name].append(self.np_random.uniform(robot.lower_limits[joint]*joint_randomness, robot.upper_limits[joint]*joint_randomness))
            #     robot.set_joint_angles(robot.arm_joint_indices, random_angles[robot_name])
            for robot_name, robot in self.my_robots.items():
                robot_pos, _ = robot.get_base_pos_orient()
                random_end_effector_pos = [random.uniform(robot_pos[0]-radius, robot_pos[0]+radius), 
                                           random.uniform(robot_pos[1]-radius, robot_pos[1]+radius),
                                           random.uniform(robot_pos[2], robot_pos[2]+radius)]
                self.set_end_effector_pos(robot, random_end_effector_pos, threshold=1e-2, maxIter=100)
                
            
            # Collect all joint pos and obj pos(last 4 joints is enough)
            joints_pos = {}
            for robot_name, robot in self.my_robots.items():
                joints_pos[robot_name] = []
                for joint in robot.arm_joint_indices[-5:]:
                    j_pos, _ = robot.get_pos_orient(joint)
                    joints_pos[robot_name].append(j_pos)
            
            objects_pos = []
            for obj in Object.instances:
                for joint in range(-1,p.getNumJoints(obj.body, physicsClientId=self.id)):
                    obj_pos, _ = obj.get_pos_orient(joint)
                    objects_pos.append(obj_pos)

            # Check for collision between robots and objects in the environment
            done = True
            for robot_name_i, robot_i in self.my_robots.items():
                for robot_name_j, robot_j in self.my_robots.items():
                    if robot_name_i != robot_name_j:
                        joints_pos_i = joints_pos[robot_name_i]
                        joints_pos_j = joints_pos[robot_name_j]
                        for joint_pos_i in joints_pos_i:
                            for joint_pos_j in joints_pos_j:
                                dist = np.linalg.norm(joint_pos_i-joint_pos_j)
                                if abs(dist) < min_dist:
                                    done = False
            
            for robot_name, robot in self.my_robots.items():
                for obj_pos in objects_pos:
                    joint_pos = joints_pos[robot_name]
                    dist = np.linalg.norm(joint_pos-dist)
                    if abs(dist) < min_dist:
                        done = False

    def set_end_effector_pos(self, robot, target_position, target_orient=None, threshold=1e-15, maxIter=1000):
        if target_orient is not None and len(target_orient) == 3:
            target_orient = self.get_quaternion(target_orient)
        closeEnough = False
        iter = 0
        dist2 = 1e30
        while (not closeEnough and iter < maxIter):
            joint_pos = p.calculateInverseKinematics(bodyIndex=robot.body, endEffectorLinkIndex=robot.end_effector, targetPosition=target_position, targetOrientation=target_orient, physicsClientId=self.id)
            robot.set_joint_angles_all(joint_pos)
            ls = p.getLinkState(robot.body, robot.end_effector)
            newPos = ls[4]
            diff = [target_position[0] - newPos[0], target_position[1] - newPos[1], target_position[2] - newPos[2]]
            dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
            closeEnough = (dist2 < threshold)
            iter = iter + 1

    def disable_collision(self, obj_1, obj_2):
        body_1 = obj_1.body
        body_2 = obj_2.body
        for i in range(p.getNumJoints(body_1, physicsClientId=self.id)):
            for j in range(p.getNumJoints(body_2, physicsClientId=self.id)):
                p.setCollisionFilterPair(body_1, body_2, i, j, 0, physicsClientId=self.id)

    def get_euler(self, quaternion):
        return np.array(p.getEulerFromQuaternion(np.array(quaternion), physicsClientId=self.id))

    def get_quaternion(self, euler):
        return np.array(p.getQuaternionFromEuler(np.array(euler), physicsClientId=self.id))

    def init_env_variables(self):
        # Select all grippable objects
        i = 0
        self.all_grippable_objects = {}
        for obj in Object.instances:
            if obj.enable_gripping:
                i += 1
                object_name = 'object_' + str(i)
                self.all_grippable_objects[object_name] = obj

        for robot_name, robot in self.my_robots.items():
            robot.buff = 0
            robot.grippable = {}
            robot.ready_to_grip = False
            for object_name, obj in self.all_grippable_objects.items():
                robot.grippable[object_name] = {'obj': obj, 'grippable': {}, 'constraint': {}}
                for joint in range(-1,p.getNumJoints(obj.body, physicsClientId=self.id)):
                    robot.grippable[object_name]['constraint']['joint_'+str(joint)] = None