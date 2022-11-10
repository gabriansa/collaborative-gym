import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
import matplotlib.pyplot as plt
import csv  
from os.path import exists

from collaborative_gym.ray_training_config import get_agent

def load_policy(env, algo, env_name, policy_path=None, seed=1):
    agent = get_agent(env, algo, env_name, seed)
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                agent.restore(checkpoint_path)
                # return agent, checkpoint_path
            return agent, None
    return agent, None

def make_env(env_name, seed=1):
    module = importlib.import_module('collaborative_gym.envs')
    env_class = getattr(module, env_name.split('-')[0] + 'Env')
    env = env_class()
    env.seed(seed)
    return env

def train(env_name, algo, timesteps_total=1000000, save_dir='./ray_trained_models/', load_policy_path='', live_graph=False, csv_bool=False, seed=1):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name)
    agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, seed)
    env.disconnect()

    file_exists = exists('learning_curve_'+env_name+'.csv')

    if file_exists:
        all_timesteps_csv =[]
        with open('learning_curve_'+env_name+'.csv') as f:
            for row in f:
                all_timesteps_csv.append(row.split(',')[0])
        last_timestep = int(all_timesteps_csv[-1])
        timesteps = last_timestep
    else:
        timesteps = 0
        last_timestep = 0

    # Live plot 
    if live_graph:
        plt.axis()
        plt.title("Learning Curves")
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        if file_exists and csv_bool:
            filename = open('learning_curve_'+env_name+'.csv','r')
            file = csv.DictReader(filename)
            all_timesteps = []
            all_reward = []
            for col in file:
                all_timesteps.append(float(col['Timestep']))
                all_reward.append(float(col['All Rewards']))
            mean_policy_reward = {}
            for robot_name, robot in env.my_robots.items():
                mean_policy_reward[robot_name] = []

                filename = open('learning_curve_'+env_name+'.csv','r')
                file = csv.DictReader(filename)
                for col in file:
                    mean_policy_reward[robot_name].append(float(col[robot_name]))
        else:
            all_timesteps = []
            all_reward = []
            mean_policy_reward = {}
            for robot_name, robot in env.my_robots.items():
                mean_policy_reward[robot_name] = []

    policy_reward_robots = {}
    for robot_name, robot in env.my_robots.items():
        policy_reward_robots[robot_name] = {}

    if csv_bool:
        if not file_exists:
            header = ['Timestep', 'All Rewards']
            for robot_name, robot in env.my_robots.items():
                header.append(robot_name)
            f = open('learning_curve_'+env_name+'.csv', 'w')
            writer = csv.writer(f)
            # write the header
            writer.writerow(header)
            # close the file
            f.close()

    while timesteps < timesteps_total:
        result = agent.train()
        timesteps = result['timesteps_total'] + last_timestep
    
        print()
        print(f"Iteration: {result['training_iteration']}, total timesteps: {timesteps}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, total mean reward (min/max): {result['episode_reward_mean']:.1f} ({result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f})")
        print("Agents' specific policy reward")
        for robot_name, robot in env.my_robots.items():
            agent_id = robot_name
            policy_reward_robots[robot_name]['mean_policy_reward'] = round(result['sampler_results']['policy_reward_mean'][agent_id],1)
            policy_reward_robots[robot_name]['min_policy_reward'] = round(result['sampler_results']['policy_reward_min'][agent_id],1)
            policy_reward_robots[robot_name]['max_policy_reward'] = round(result['sampler_results']['policy_reward_max'][agent_id],1)
            print(str(agent_id) + "--> " + "mean :" + str(policy_reward_robots[robot_name]['mean_policy_reward']) + " min/max: " + str(policy_reward_robots[robot_name]['min_policy_reward']) + "/" + str(policy_reward_robots[robot_name]['max_policy_reward']))
        sys.stdout.flush()  

        # Delete the old saved policy
        if checkpoint_path is not None:
            shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        # Save the recently trained policy
        checkpoint_path = agent.save(os.path.join(save_dir, algo, env_name))

        # Live plot
        if live_graph:
            plt.close()
            all_timesteps.append(timesteps)
            reward_mean = result['episode_reward_mean']
            all_reward.append(reward_mean)
            plt.plot(all_timesteps, all_reward, 'b-', label='all robots')
            for robot_name, robot in env.my_robots.items():
                try:
                    mean_policy_reward[robot_name].append(policy_reward_robots[robot_name]['mean_policy_reward'])
                except:
                    mean_policy_reward[robot_name].append(np.nan)
                plt.plot(all_timesteps, mean_policy_reward[robot_name], '-', label=robot_name)
            plt.legend()
            plt.pause(0.5)

        if csv_bool:
            # Write csv file
            f = open('learning_curve_'+env_name+'.csv', 'a')
            writer = csv.writer(f)
            # write the header
            csv_data = [timesteps, result['episode_reward_mean']]
            for robot_name, robot in env.my_robots.items():
                csv_data.append(policy_reward_robots[robot_name]['mean_policy_reward'])
            writer.writerow(csv_data)
            # close the file
            f.close()

    return checkpoint_path

def render_policy(env, env_name, algo, policy_path, seed=1, n_episodes=1):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, seed=seed)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, seed)


    env.render()
    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action_robots = {}
            for robot_name, robot in env.my_robots.items():
                action_robots[robot_name] = test_agent.compute_action(obs[robot_name], policy_id=robot_name)

            obs, reward, done, info = env.step(action_robots)
            done = done['__all__']

    env.disconnect()

def evaluate_policy(env_name, algo, policy_path, n_episodes=100, seed=1):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, seed=seed)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, seed)

    rewards = []
    task_performances = []
    task_completions = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        reward_total = 0.0
        task_success = 0.0
        while not done:
            action_robots = {}
            # Compute the next action for the robots using the trained policies
            for robot_name, robot in env.my_robots.items():
                action_robots[robot_name] = test_agent.compute_action(obs[robot_name], policy_id=robot_name)

            # Step the simulation forward using the actions from the trained policies
            obs, reward, done, info = env.step(action_robots)
            for robot_name, robot in env.my_robots.items():
                reward_total += reward[robot_name]
                # task_success += info[robot_name]['task_success']
            done = done['__all__']
        for robot_name, robot in env.my_robots.items():
            task_performances.append(info[robot_name]['task_performance_%'])
            task_completions.append(info[robot_name]['task_completion'])
            # task_successes.append(task_success/200)
        rewards.append(reward_total)
        sys.stdout.flush()
    env.disconnect()

    print('\n', '-'*50, '\n')
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))

    # print('Task Successes:', task_successes)
    print('Task Performance Mean:', np.mean(task_performances))
    print('Task Performance Std:', np.std(task_performances))

    print('Task Completion Mean:', np.mean(task_completions))
    print('Task Completion Std:', np.std(task_completions))
    print('\n', '-'*50, '\n')
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reinforcement Learning Utility for Collaborative Gym')
    parser.add_argument('--env', default='LiftTask-v0',
                        help='Environment to train on (default: LiftTask-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm from ray[rllib]')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Used to train new policies')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Used to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Used to evaluate trained policies over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./ray_trained_models/',
                        help='Directory to save trained policy in (default ./ray_trained_models/)')
    parser.add_argument('--load-policy-path', default='./ray_trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--graph', action='store_true', default=False,
                        help='Whether training should show a live graph of the learning curve')
    parser.add_argument('--csv', action='store_true', default=False,
                        help='Whether training should save a csv file of the learning curve')
    args = parser.parse_args()

    checkpoint_path = None

    if args.train:
        checkpoint_path = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, load_policy_path=args.load_policy_path, live_graph=args.graph, csv_bool=args.csv, seed=args.seed)
    if args.render:
        render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, seed=args.seed, n_episodes=args.render_episodes)
    if args.evaluate:
        evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, seed=args.seed)