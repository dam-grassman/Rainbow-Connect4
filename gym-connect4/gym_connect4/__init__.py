from gym.envs.registration import register
from gym_connect4.envs.connect4_env import connect4Env
#from gym_connect4.envs.connect4_extrahard_env import connect4ExtraHardEnv

register(
    id='connect4-v0',
    entry_point='gym_connect4.envs:connect4Env',
)
register(
    id='connect4-extrahard-v0',
    entry_point='gym_connect4.envs:connect4ExtraHardEnv',
)