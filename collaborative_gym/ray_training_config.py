import multiprocessing
from ray.rllib.agents import ppo, sac, ddpg

def setup_config(env, algo, seed=0):
    num_processes = multiprocessing.cpu_count()
    print("Using "+str(num_processes)+" cores")

     # === PPO Configurations ===
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 19200
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 128
        config['lambda'] = 0.95
        # config['gamma'] = 0.8
        config['model']['fcnet_hiddens'] = [256, 256]
        config['model']['fcnet_activation'] = "relu" # Supported values are: "tanh", "relu", "swish" (or "silu"),"linear" (or None).
    
    config['framework'] = 'tf2'
    config['eager_tracing'] = True

    # Setup number of agents and their policies
    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'

    obs = env.reset()
    policies = {}
    for robot_name, robot in env.my_robots.items():
        policies[robot_name] = (None, env.observation_space_robot[robot_name], env.action_space_robot[robot_name], {})
    
    # config['multiagent'] = {'policies': policies, 'policy_mapping_fn': lambda a: a}
    config['multiagent'] = {'policies': policies, 'policy_mapping_fn': (lambda agent_id, episode, **kwargs: agent_id)}
    config['env_config'] = {'num_agents': len(env.my_robots)}
    
    return {**config}

def get_agent(env, algo, env_name, seed=0):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, seed), 'collaborative_gym:'+env_name)
    return agent