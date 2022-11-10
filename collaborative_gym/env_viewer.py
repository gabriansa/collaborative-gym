import gym, sys, argparse
import numpy as np

if sys.version_info < (3, 0):
    print('Please use Python >=3.8.10')
    exit()

def sample_action(env):
    action = {}
    for robot_name, robot in env.my_robots.items():
        action[robot_name] = env.action_space_robot[robot_name].sample()
    return action

def viewer(env_name):
    env = gym.make(env_name)

    while True:
        done = False
        env.render()
        observation = env.reset()
        action = sample_action(env)
       
        for robot_name, robot in env.my_robots.items():
            print(robot_name + ' ('+type(robot).__name__+')' + ' --> ' + 'Observation size:', np.shape(observation[robot_name]), 'Action size:', np.shape(action[robot_name]))

        while not done:
            observation, reward, done, info = env.step(sample_action(env))
            if type(done) is dict:
                done = done['__all__']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collaborative Gym Environment Viewer')
    parser.add_argument('--env', default='LiftTask-v0',
                        help='Default Environment: LiftTask-v0')
    args = parser.parse_args()

    viewer(args.env)