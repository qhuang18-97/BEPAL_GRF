import copy
from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
import os
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value','value_g', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'system_reward','misc','node_maploss' ))#,'l0','l1','l2',


class Trainer(object):
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = False
        self.last_step = False
        self.optimizer = optim.RMSprop(policy_net.parameters(),
            lr = args.lrate, alpha=0.97, eps=1e-6)
        self.params = [p for p in self.policy_net.parameters()]
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=40000, gamma=0.3)
        #self.device = torch.device('cpu') #'cuda:0' if torch.cuda.is_available else

    def ground_truth_gen(self, obs, action):
        pos = []
        pos.extend((obs['left_team']).flatten())
        pos.extend((obs['right_team']).flatten())
        nnodes = obs['left_team'].shape[0]-1 + obs['right_team'].shape[0]+1
        # ball position
        pos.extend(obs['ball'])
        node = np.array(pos[2:-1]).reshape(nnodes, 2)
        #node[:,0] = node[:,0] /2
        #node[:, 1] = node[:, 1] / 0.84
        node = np.concatenate((node, np.zeros([ nnodes,1])), axis=1)
        node[-1,-1] = obs['ball'][-1]
        # ball direction
        direction = []
        direction.extend(obs['left_team_direction'].flatten())
        direction.extend(obs['right_team_direction'].flatten())
        direction.extend(obs['ball_direction'])
        supp_dir = np.array(direction[2:-1]).reshape( nnodes, 2)
        node = np.concatenate((node, supp_dir, np.zeros([nnodes,1])), axis=1)
        node[-1,-1] = obs['ball_direction'][-1]
        ot = []
        if obs['ball_owned_team'] == -1:
            ot.extend([1, 0, 0])

        if obs['ball_owned_team'] == 0:
            ot.extend([0, 1, 0])

        if obs['ball_owned_team'] == 1:
            ot.extend([0, 0, 1])

        id = np.zeros([self.args.nagents,nnodes,2])
        for a in range(id.shape[0]):
            for b in range(id.shape[1]):
                if a == b:
                    id[a, b, 1] = 1
                if b < 3:
                    id[a, b, 0] = 1
        node = np.concatenate((node, np.array(nnodes*ot).reshape(nnodes, 3)), axis=1)
        pos_map = np.concatenate((np.array(pos[2:-1]).reshape(nnodes, 2), np.zeros([nnodes,1])), axis=1)
        pos_map[-1,-1] = obs['ball'][-1]
        node = node[np.newaxis].repeat(self.args.nagents, axis=0)
        node = np.concatenate((node, id), axis=2)
        # generate edge feature ground truth
        edge = np.zeros([nnodes,nnodes])
        att_list = list(range(self.args.nagents))
        att_list.append(nnodes-1)
        for i in range(nnodes):
            for j in range(nnodes):
                if i in att_list or j in att_list:
                    edge[i,j] = np.sqrt(np.sum((pos_map[i] - pos_map[j]) ** 2))
        # node = node[:, :, :-2]  # Now shape is [num_agents, nnodes, 8]
        # node = np.concatenate((node[:, :, 0:4], node[:, :, 6:]), axis=-1)
        # node = np.concatenate((node[:, :, 0:7], node[:, :, 10:]), axis=-1)
        return node, edge

    def graph_extract(self, obs, glb):
        obs = np.asarray(torch.squeeze(obs))
        # nnodes = (np.array(obs).shape[1]-10-3)//2 # for GAT Without GK
        nnodes = (np.array(obs).shape[1] - 10 - 1) // 2  # for GAT Without GK
        nodes = np.zeros([self.args.nagents, nnodes, 3+3])
        adj = np.zeros([self.args.nagents, nnodes, nnodes])
        obs = obs[:, 2:-7]
        # obs = obs[:, :-7]
        mask = np.ones([self.args.nagents,nnodes])
        id = np.zeros([self.args.nagents,nnodes,2])
        for z in range(self.args.nagents):
            for x,y in enumerate(list(obs[z, :-3])):
                if y==-20:
                    if x == obs[z, :-3].shape[0]-1:
                        mask[z,(x//2)-1] = 0
                    else:
                        mask[z,x//2] = 0
        for a in range(id.shape[0]):
            for b in range(id.shape[1]):
                if a==b:
                    id[a, b, 1] = 1
                if b<3:
                    id[a, b, 0] = 1
        idx = 0
        for i in range(nodes.shape[1]):
            nodes[:, i, 0:2] = obs[:, idx:idx+2]
            nodes[:, i, -3:] = obs[:, -3:]
            if i == nodes.shape[1]-1:
                nodes[:, i, 2] = obs[:, idx]
            idx = idx +2
        node_before_mask = np.concatenate((nodes,id), axis=2)
        # node = np.delete(node_before_mask, np.where(mask == 0)[0], axis=0)
        node_matirx = []
        adj_clean =[]
        for i in range(mask.shape[0]):
            adj[i, i, :] = mask[i, :]
            adj[i, :, i] = mask[i, :].T
            # adj[i, i, i] = 0
            adj_temp = np.delete(adj[i],np.where(~adj[i].any(axis=0))[0], axis=1)
            adj_temp = adj_temp[~np.all(adj_temp == 0, axis=1)]
            node_matirx.append(torch.tensor(np.delete(node_before_mask[i], np.where(mask[i] == 0)[0], axis=0)))
            adj_clean.append(torch.tensor(adj_temp))
        return node_matirx, adj_clean


    def get_episode(self, epoch, episodes):
        # path = './50case/ep' + str(episodes)
        # if not os.path.exists(path):
        #       os.mkdir(path)
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info = dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)

        for t in range(self.args.max_steps):
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info['comm_action'] = np.zeros(self.args.nagents, dtype=int)

            # recurrence over time
            if self.args.recurrent:
                if self.args.rnn_type == 'LSTM' and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                # node, adj = self.graph_extract(state, self.env.env.global_info)
                # x = [node, adj, prev_hid]
                x = [state, prev_hid]
                #if np.any(info['comm_action'] == 1):
                    #checkpoint = x
                action_out, value, value_g,prev_hid, decoded_node= self.policy_net(x, info) #
                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())

                    else:
                        prev_hid = prev_hid.detach()

            else:
                x = state
                action_out, value ,value_g= self.policy_net(x, info) #

            action = select_action(self.args, action_out)
            action, actual = translate_action(self.args, self.env, action)
            #### for game without control GK self.nagents+self.args.right_players+3, otherwise +2
            agent_map = decoded_node.view(self.args.nagents,(self.args.nagents+self.args.right_players+2),-1)


            node_gt, edge_gt = self.ground_truth_gen(self.env.env.global_info,actual[0])
            # gt = gt[np.newaxis]
            # node_ground_truthg = torch.tensor(node_gt[np.newaxis].repeat(3, axis=0))
            #c_node = agent_map.detach().numpy()
            #c_edge = edge_map.detach().numpy()
            #np.save('data/node_decode_' + 'episode'+str(episodes)+'_step'+str(t) + '.npy', c_node)
            #np.save('data/edge_decode_' + 'episode' + str(episodes) + '_step' + str(t) + '.npy', c_edge)
            #np.save('data/edge_gt_' + 'episode' + str(episodes) + '_step' + str(t) + '.npy', edge_gt)
            #np.save('data/node_gt_' + 'episode' + str(episodes) + '_step' + str(t) + '.npy', node_gt)
            node_ground_truthg = torch.tensor(node_gt)

            Loss_func = nn.MSELoss(reduction='sum')
            node_maploss = Loss_func(agent_map, node_ground_truthg.detach())

            vis_grth =  np.array(node_ground_truthg.detach())
            vis_decode = np.array(agent_map.detach())

            #np.save(path+'/global_s'+str(t)+'.npy', self.env.env.global_info)
            #np.save(path + '/action_s' + str(t) + '.npy', actual)
            next_state, reward, done, info = self.env.step(actual)
            reward, system_reward = reward
            # add sys reward to reward function
            if done and t == self.args.max_steps - 1:
                system_reward += -0.2

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                info['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)

                stat['comm_action'] = stat.get('comm_action', 0) + info['comm_action'][:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()


            trans = Transition(state, action, action_out, value, value_g, episode_mask, episode_mini_mask, next_state, reward,system_reward, misc, node_maploss)
            #trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc, maploss)

            episode.append(trans)
            state = next_state
            if done:
                #np.save(path + '/global_s' + str(t) + '.npy', self.env.env.global_info)
                #np.save(path + '/action_s' + str(t) + '.npy', actual)
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']
        ############# sons data
        # np.save('./data/trajectory_map'+str(episodes)+'.npy', tracj)
        # np.save('./data/communication_history_map'+str(episodes)+'.npy', comm_history)

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        # stat['min_steps'] = self.env.env.min_steps # pretrain vision
        # stat['min_steps'] = 0 # pretrain vision 2
        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return (episode, stat)

    def compute_grad(self, batch):
        stat = dict()
        num_actions = self.args.num_actions
        dim_actions = self.args.dim_actions

        n = self.args.nagents
        batch_size = len(batch.state)

        rewards = torch.Tensor(batch.reward)
        system_rewards = torch.Tensor(batch.system_reward)
        episode_masks = torch.Tensor(batch.episode_mask)
        episode_mini_masks = torch.Tensor(batch.episode_mini_mask)
        actions = torch.Tensor(batch.action)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)
        '''
        rewards = rewards.to(self.device)
        episode_masks = episode_masks.to(self.device)
        episode_mini_masks = episode_mini_masks.to(self.device)
        actions = actions.to(self.device)
        '''
        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0)
        values_g = torch.cat(batch.value_g, dim=0)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1)
        #alive_masks = alive_masks.to(self.device)
        coop_returns = torch.Tensor(batch_size, n)#.cuda()
        ncoop_returns = torch.Tensor(batch_size, n)#.cuda()
        returns = torch.Tensor(batch_size, n)#.cuda()
        deltas = torch.Tensor(batch_size, n)#.cuda()
        advantages = torch.Tensor(batch_size, n)#.cuda()
        values = values.view(batch_size, n)
        values_g = values_g.view(batch_size, n,n) #

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0

        sys_coop_returns = torch.Tensor(batch_size, n)  # .cuda()
        sys_ncoop_returns = torch.Tensor(batch_size, n)  # .cuda()
        sys_prev_coop_return = 0
        sys_prev_ncoop_return = 0
        sys_returns = torch.Tensor(batch_size, n)
        #sys_advantages = torch.Tensor(batch_size, n)

        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])
            '''self.args.gamma   system_system_'''
            sys_coop_returns[i] = rewards[i] + 0.95 * sys_prev_coop_return * episode_masks[i]
            sys_ncoop_returns[i] = rewards[i] + 0.95 * sys_prev_ncoop_return * episode_masks[i] * \
                                   episode_mini_masks[i]

            sys_prev_coop_return = sys_coop_returns[i].clone()
            sys_prev_ncoop_return = sys_ncoop_returns[i].clone()

            sys_returns[i] = (self.args.mean_ratio * sys_coop_returns[i].mean()) \
                             + ((1 - self.args.mean_ratio) * sys_ncoop_returns[i])


        for i in reversed(range(rewards.size(0))):
            advantages[i] = returns[i] - values.data[i]
            # sys_advantages[i] = sys_returns[i] - values_g.data[i]

        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        map_loss_0 = torch.stack(batch.node_maploss, dim=0)

        map_loss_m0 = map_loss_0.sum()


        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()
        check_rwd = np.array(rewards)
        check_rtn = np.array(returns)
        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()
        # gloable value loss term
        '''targets_g = sys_returns.sum(1).view(batch_size, 1)
        
        
        # targets_g = returns.sum(1).view(batch_size, 1)
        value_loss_g = (values_g - targets_g).pow(2).view(-1) #/(self.args.nagents)
        value_loss_g *= alive_masks
        value_loss_g = value_loss_g.sum()
        '''
        targets_g = sys_returns.unsqueeze(1).repeat(1, n, 1)
        # value_loss_g = (values_g/self.args.nagents - targets_g/self.args.nagents).pow(2).view(-1)
        value_loss_g = (values_g - targets_g).pow(2).view(-1)  # Feb setting
        value_loss_g *= alive_masks.repeat(n).view(-1)  #
        value_loss_g = value_loss_g.sum()
        stat['value_loss'] = value_loss.item()
        stat['value_loss_g'] =  (value_loss_g/self.args.nagents).item()

        map_loss = (map_loss_m0 / (n + 3) ) #+ map_loss_m1 / (n + 3)#edge comment: ###############(map_loss_m0 / ((n + 3) * (n + 5)) + map_loss_m1 / (n + 3) ** 2)
        stat['map_loss'] =  map_loss.item()# (map_loss_m0/ ((n+3)*(n+5)) + map_loss_m1/(n+3)**2).item()

        loss = action_loss + self.args.value_coeff * (value_loss )  + 0.5*map_loss#### '''+ self.args.value_coeff *(value_loss_g/(self.args.nagents)) +
        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy


        stat['loss'] = loss.item()
        loss.backward()

        return stat

    def run_batch(self, epoch):
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        # while self.stats['num_episodes'] < 50:
        while len(batch) < self.args.batch_size:  # commended for data collection
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch, self.stats['num_episodes'])
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        batch, stat = self.run_batch(epoch)
        self.optimizer.zero_grad()

        s = self.compute_grad(batch)
        merge_stat(s, stat)
        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()

        return stat

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
