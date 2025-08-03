import copy

import gym
import gfootball.env as grf_env
import numpy as np
import torch


class GRFWrapperEnv(gym.Env):
    
    def __init__(self,):
        self.env = None
        self.num_controlled_lagents = 0
        self.num_controlled_ragents = 0
        self.num_controlled_agents = 0
        self.num_lagents = 0
        self.num_ragents = 0
        self.action_space = None  # gym.spaces.Discrete(19)
        self.observation_space = None  # gym.spaces.Box(low=-1,high=1,dtype=np.float32)
        
    def init_args(self, parser):
        env = parser.add_argument_group('GRF')
        env.add_argument('--scenario', type=str, default='academy_3_vs_1_with_keeper',
                         help="Scenario of the game")        
        env.add_argument('--num_controlled_lagents', type=int, default=3,
                         help="Number of controlled agents on the left side")
        env.add_argument('--num_controlled_ragents', type=int, default=0,
                         help="Number of controlled agents on the right side")  
        env.add_argument('--reward_type', type=str, default='scoring',
                         help="Reward type for training")
        env.add_argument('--render', action="store_true", default=False,
                         help="Render training or testing process")
        
    def multi_agent_init(self, args):
        self.env = grf_env.create_environment(
            env_name=args.scenario,
            stacked=False,
            representation='multiagent',
            rewards=args.reward_type,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=args.render,
            dump_frequency=0,
            logdir='/tmp/test',
            extra_players=None,
            number_of_left_players_agent_controls=args.num_controlled_lagents,
            number_of_right_players_agent_controls=args.num_controlled_ragents,
            channel_dimensions=(3, 3))
        self.num_controlled_lagents = args.num_controlled_lagents
        self.num_controlled_ragents = args.num_controlled_ragents
        self.num_controlled_agents = args.num_controlled_lagents + args.num_controlled_ragents
        self.num_lagents = self.env.num_lteam_players
        self.num_ragents = self.env.num_rteam_players
        if self.num_controlled_agents > 1:
            action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        else:
            action_space = self.env.action_space
        if self.num_controlled_agents > 1:
            observation_space = gym.spaces.Box(
                low=self.env.observation_space.low[0],
                high=self.env.observation_space.high[0],
                dtype=self.env.observation_space.dtype)
        else:
            observation_space = gym.spaces.Box(
                low=self.env.observation_space.low,
                high=self.env.observation_space.high,
                dtype=self.env.observation_space.dtype)
            
        # check spaces
        self.action_space = action_space
        self.observation_space = observation_space
        self.global_info = dict()
        self.partial_mask = np.ones([3, 25])
        self.ball_owned_player = -1
        self.last_ball_x = -1
        self.global_state = np.zeros([3, 25])
        return
   
    # check epoch arg
    def reset(self):
        self.stat = dict()
        obs = self.env.reset()
        self.global_state = obs
        self.global_info = dict()
        self.partial_mask = np.ones([obs.shape[0], obs.shape[1]])
        if self.num_controlled_agents == 1:
            obs = obs.reshape(1, -1)
        self.global_info = self.env.env.unwrapped._env._observation
        self.last_ball_x = self.global_info['ball'][0]
        return obs
    
    def step(self, actions):
        self.last_ball_x = self.global_info['ball'][0]
        o, r, d, i = self.env.step(actions) # [0] for bams
        self.global_info = self.env.env.unwrapped._env._observation
        o = self.partial_obs(o)
        if self.num_controlled_agents == 1:
            o = o.reshape(1, -1)
            r = r.reshape(1, -1)
        next_obs = o
        system_reward = self.reward_sys(r)
        rewards = (r, system_reward)
        dones = d
        infos = i
        self.stat['success'] = infos['score_reward']
        return next_obs, rewards, dones, infos

    def partial_obs(self, o):
        agent_loc = []
        left = self.global_info['left_team'].shape[0]
        right = self.global_info['right_team'].shape[0]
        for i in range(left):
            agent_loc.append(self.global_info['left_team'][i].tolist())
        for i in range(right):
            agent_loc.append(self.global_info['right_team'][i].tolist())
        agent_loc.append(self.global_info['ball'][0:2].tolist())
        mask = np.ones([left + right + 1, left + right + 1])
        for i in range(left + right + 1):
            for j in range(left + right + 1):
                d = np.sqrt((agent_loc[i][0] - agent_loc[j][0]) ** 2 + (agent_loc[i][1] - agent_loc[j][1]) ** 2)
                if d > 0.5:
                    mask[i][j] = -1e5
        mask = mask[1:left] # 1:left for game without GK, :left for game with GK as controalable agent
        expand_mask = np.ones([self.num_controlled_agents, o.shape[1]])
        expand_mask[:, :(left + right + 1)*2] = np.repeat(mask, 2, axis=1)
        self.partial_mask = expand_mask
        obs = o * expand_mask
        obs[obs > 20] = -20
        obs[obs < -20] = -20
        return obs


    def reward_sys(self, r):
        r_sys = copy.deepcopy(r)
        if self.ball_owned_player != self.global_info['ball_owned_player']:
            if self.global_info['ball_owned_team'] == 1:
                r_sys[self.ball_owned_player-1] -= 0.5
                r_sys -= 0.05
                self.ball_owned_player = self.global_info['ball_owned_player']
            if self.global_info['ball_owned_team'] == 0:
                # directly use ball position as reward
                r_sys += 0.01 * self.global_info['ball'][0]
                r_sys[self.global_info['ball_owned_player']-1] += 0.09 * self.global_info['ball'][0]
                # using ball offset as reward
                # r_sys += 0.01 * max((self.global_info['ball'][0] - self.last_ball_x), 0)
                # r_sys[self.global_info['ball_owned_player'] - 1] += 0.09 * max((self.global_info['ball'][0] - self.last_ball_x), 0)

                self.ball_owned_player = self.global_info['ball_owned_player']
        else:
            if self.ball_owned_player > 0:
                # directly use ball position as reward
                r_sys += 0.01 * self.global_info['ball'][0]
                r_sys[self.ball_owned_player-1] += 0.09 * self.global_info['ball'][0]
                # using ball offset as reward
                # r_sys += 0.01 * max((self.global_info['ball'][0] - self.last_ball_x), 0)
                # r_sys[self.global_info['ball_owned_player'] - 1] += 0.09 * max((self.global_info['ball'][0] - self.last_ball_x), 0)
        return r_sys

    def seed(self):
        return
    
    def render(self):
        self.env.render()
        
    def exit_render(self):
        self.env.disable_render()

    # def truth_fig(self):
