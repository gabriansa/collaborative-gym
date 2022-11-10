import numpy as np
import pybullet as p

class Agent:
    def __init__(self):
        self.base = -1
        self.body = None
        self.lower_limits = None
        self.upper_limits = None
        self.ik_lower_limits = None
        self.ik_upper_limits = None
        self.ik_joint_names = None

    def init_env(self, body, env, indices=None):
        self.init(body, env.id, env.np_random, indices)

    def init(self, body, id, np_random, indices=None):
        self.body = body
        self.id = id
        self.np_random = np_random
        self.all_joint_indices = list(range(p.getNumJoints(body, physicsClientId=id)))
        if indices != -1:
            self.update_joint_limits()
            self.enforce_joint_limits(indices)
            self.controllable_joint_lower_limits = np.array([self.lower_limits[i] for i in self.controllable_joint_indices])
            self.controllable_joint_upper_limits = np.array([self.upper_limits[i] for i in self.controllable_joint_indices])

    def control(self, indices, target_angles, gains, forces):
        if type(gains) in [int, float]:
            gains = [gains]*len(indices)
        if type(forces) in [int, float]:
            forces = [forces]*len(indices)
        p.setJointMotorControlArray(self.body, jointIndices=indices, controlMode=p.POSITION_CONTROL, targetPositions=target_angles, positionGains=gains, forces=forces, physicsClientId=self.id)
        
    def get_joint_angles(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        elif not indices:
            return []
        robot_joint_states = p.getJointStates(self.body, jointIndices=indices, physicsClientId=self.id)
        return np.array([x[0] for x in robot_joint_states])

    def get_joint_angles_dict(self, indices=None):
        return {j: a for j, a in zip(indices, self.get_joint_angles(indices))}

    def get_pos_orient(self, link, center_of_mass=False):
        # Get the 3D position and orientation (4D quaternion) of a specific link on the body
        if link == self.base:
            pos, orient = p.getBasePositionAndOrientation(self.body, physicsClientId=self.id)
        else:
            if not center_of_mass:
                pos, orient = p.getLinkState(self.body, link, computeForwardKinematics=True, physicsClientId=self.id)[4:6]
            else:
                pos, orient = p.getLinkState(self.body, link, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        return np.array(pos), np.array(orient)

    def get_finger_COM(self):
        finger_COM_pos, finger_COM_orient = self.get_pos_orient(self.end_effector)
        return np.array(finger_COM_pos), np.array(finger_COM_orient)

    def get_base_pos_orient(self):
        return self.get_pos_orient(self.base)

    def get_euler(self, quaternion):
        return np.array(p.getEulerFromQuaternion(np.array(quaternion), physicsClientId=self.id))

    def get_quaternion(self, euler):
        return np.array(p.getQuaternionFromEuler(np.array(euler), physicsClientId=self.id))

    def get_joint_max_force(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        joint_infos = [p.getJointInfo(self.body, i, physicsClientId=self.id) for i in indices]
        return [j[10] for j in joint_infos]

    def set_base_pos_orient(self, pos, orient):
        p.resetBasePositionAndOrientation(self.body, pos, orient if len(orient) == 4 else self.get_quaternion(orient), physicsClientId=self.id)

    def set_joint_angles_all(self, angles):
        i = 0
        for joint in range(p.getNumJoints(self.body, physicsClientId=self.id)):
            if p.getJointInfo(self.body, joint, physicsClientId=self.id)[2] != 4:
                p.resetJointState(self.body, joint, angles[i])
                i += 1

    def set_joint_angles(self, indices, angles, use_limits=True, velocities=0):
        for i, (j, a) in enumerate(zip(indices, angles)):
            p.resetJointState(self.body, jointIndex=j, targetValue=min(max(a, self.lower_limits[j]), self.upper_limits[j]) if use_limits else a, targetVelocity=velocities if type(velocities) in [int, float] else velocities[i], physicsClientId=self.id)

    def set_gravity(self, ax=0.0, ay=0.0, az=-9.81):
        p.setGravity(ax, ay, az, body=self.body, physicsClientId=self.id)

    def update_joint_limits(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        self.lower_limits = dict()
        self.upper_limits = dict()
        self.ik_lower_limits = []
        self.ik_upper_limits = []
        self.ik_joint_names = []
        for j in indices:
            joint_info = p.getJointInfo(self.body, j, physicsClientId=self.id)
            joint_name = joint_info[1]
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if lower_limit == 0 and upper_limit == -1:
                lower_limit = -1e10
                upper_limit = 1e10
                if joint_type != p.JOINT_FIXED:
                    # NOTE: IK only works on non fixed joints, so we build special joint limit lists for IK
                    self.ik_lower_limits.append(-2*np.pi)
                    self.ik_upper_limits.append(2*np.pi)
                    self.ik_joint_names.append([len(self.ik_joint_names)] + list(joint_info[:2]))
            elif joint_type != p.JOINT_FIXED:
                self.ik_lower_limits.append(lower_limit)
                self.ik_upper_limits.append(upper_limit)
                self.ik_joint_names.append([len(self.ik_joint_names)] + list(joint_info[:2]))
            self.lower_limits[j] = lower_limit
            self.upper_limits[j] = upper_limit
        self.ik_lower_limits = np.array(self.ik_lower_limits)
        self.ik_upper_limits = np.array(self.ik_upper_limits)

    def enforce_joint_limits(self, indices=None):
        if indices is None:
            indices = self.all_joint_indices
        joint_angles = self.get_joint_angles_dict(indices)
        if self.lower_limits is None or len(indices) > len(self.lower_limits):
            self.update_joint_limits()
        for j in indices:
            if joint_angles[j] < self.lower_limits[j]:
                p.resetJointState(self.body, jointIndex=j, targetValue=self.lower_limits[j], targetVelocity=0, physicsClientId=self.id)
            elif joint_angles[j] > self.upper_limits[j]:
                p.resetJointState(self.body, jointIndex=j, targetValue=self.upper_limits[j], targetVelocity=0, physicsClientId=self.id)

    def inverse_kinematics(self, target_joint, target_pos, target_orient, max_iterations=1000, half_range=False, use_current_as_rest=False):
        ik_indices = self.arm_ik_indices
        if target_orient is not None and len(target_orient) < 4:
            target_orient = self.get_quaternion(target_orient)
        ik_lower_limits = self.ik_lower_limits
        ik_upper_limits = self.ik_upper_limits
        ik_joint_ranges = ik_upper_limits - ik_lower_limits
        if half_range:
            ik_joint_ranges /= 2.0
        if use_current_as_rest:
            ik_rest_poses = np.array(self.get_motor_joint_states()[1])
        else:
            ik_rest_poses = self.np_random.uniform(ik_lower_limits, ik_upper_limits)

        if target_orient is not None:
            ik_joint_poses = np.array(p.calculateInverseKinematics(self.body, target_joint, targetPosition=target_pos, targetOrientation=target_orient, lowerLimits=ik_lower_limits.tolist(), upperLimits=ik_upper_limits.tolist(), jointRanges=ik_joint_ranges.tolist(), restPoses=ik_rest_poses.tolist(), maxNumIterations=max_iterations, physicsClientId=self.id))
        else:
            ik_joint_poses = np.array(p.calculateInverseKinematics(self.body, target_joint, targetPosition=target_pos, lowerLimits=ik_lower_limits.tolist(), upperLimits=ik_upper_limits.tolist(), jointRanges=ik_joint_ranges.tolist(), restPoses=ik_rest_poses.tolist(), maxNumIterations=max_iterations, physicsClientId=self.id))
        
        for i in range(len(ik_joint_poses)):
            p.resetJointState(self.body, i, ik_joint_poses[i])
        return ik_joint_poses[ik_indices]