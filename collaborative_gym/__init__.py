import itertools
from gym.envs.registration import register
from ray.tune.registry import register_env

# Import environments
from collaborative_gym.envs.passing_sawyers import PassTaskSawyersEnv
from collaborative_gym.envs.passing_jaco_sawyer import PassTaskJacoSawyerEnv

from collaborative_gym.envs.poking_sawyers import PokeTaskSawyersEnv
from collaborative_gym.envs.poking_panda_sawyer import PokeTaskPandaSawyerEnv

from collaborative_gym.envs.lifting_sawyers import LiftTaskSawyersEnv
from collaborative_gym.envs.lifting_jacos import LiftTaskJacosEnv

from collaborative_gym.envs.balancing_sawyers import BalanceTaskSawyersEnv
from collaborative_gym.envs.balancing_jaco_sawyer import BalanceTaskJacoSawyerEnv


tasks = ['PassTaskSawyers', 'PassTaskJacoSawyer', 'PokeTaskSawyers', 'PokeTaskPandaSawyer', 'LiftTaskSawyers', 'LiftTaskJacos', 'BalanceTaskSawyers', 'BalanceTaskJacoSawyer']
tasksEnv = [PassTaskSawyersEnv, PassTaskJacoSawyerEnv, PokeTaskSawyersEnv, PokeTaskPandaSawyerEnv, LiftTaskSawyersEnv, LiftTaskJacosEnv, BalanceTaskSawyersEnv, BalanceTaskJacoSawyerEnv]

# Register environments
for task, taskEnv in zip(tasks, tasksEnv):
    id = '%s-v0' % (task)
    register(
        id= id,
        entry_point='collaborative_gym.envs:%sEnv' % (task),
        max_episode_steps=200,
    )
    # Register environment for ray rllib
    register_env('collaborative_gym:%s' % (id), lambda config: taskEnv())