# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
            self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x 
  
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 512)
        self.layer_5 = nn.Linear(512, 256)
        self.layer_6 = nn.Linear(256, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        # Forward-Propagation on the first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # Forward-Propagation on the second Critic Neural Network
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)        
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1

class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action
        self.last_state = np.zeros((5, ), dtype=np.float32)
        self.last_actions = np.zeros((3, ), dtype=np.float32)
        self.steps_since_last_train = 0
        self.total_time_steps = 0
        if os.path.isfile('last_actor.pth') and os.path.isfile('last_critic.pth'):
            self.model_loaded = 1
        else:
            self.model_loaded = 0
        print(f'self.model_loaded : {self.model_loaded}')

        self.train_iterations = 10000
        self.batch_size = 100
        self.discount = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.9

    def select_action(self, reward, state, done):
        if self.total_time_steps > 1:   
            self.replay_buffer.add((self.last_state, state.flatten(), self.last_actions, reward, done))             
        # if self.total_time_steps == 10_000 and self.model_not_loaded:                      
        #     self.train_iterations = 100 
        #     self.batch_size = 100
        #     print(f' Training at {self.total_time_steps} for {self.train_iterations}')
        #     self.train() 
        #     self.steps_since_last_train = 0
        # print(f'self.total_time_steps : {self.total_time_steps} & self.steps_since_last_train : {self.steps_since_last_train}')

        if self.model_loaded:
            if self.steps_since_last_train == 500: 
                self.train_iterations = 4
                self.batch_size = 50
                self.steps_since_last_train = 0
                print(f' self.model_not_loaded - {self.model_loaded} - Training at {self.total_time_steps} for {self.train_iterations}')
                self.train()               
        else:
            if self.total_time_steps == 10_000:
                self.train_iterations = 100 
                self.batch_size = 100
                print(f' self.model_not_loaded - {self.model_loaded} - Training at {self.total_time_steps} for {self.train_iterations}')    
                self.train() 
                self.steps_since_last_train = 0               
            if self.total_time_steps > 10_000 and self.steps_since_last_train == 500:
                self.train_iterations = 4
                self.batch_size = 50
                self.steps_since_last_train = 0
                print(f' self.model_not_loaded - {self.model_loaded} - Training at {self.total_time_steps} for {self.train_iterations}')
                self.train()            

        # if self.total_time_steps > 10_000 and self.model_not_loaded:            
        #     if self.steps_since_last_train == 500:
        #         self.train_iterations = 4
        #         self.batch_size = 50
        #         self.steps_since_last_train = 0
        #         print(f' Training at {self.total_time_steps} for {self.train_iterations}')
        #         self.train()
        
        state = torch.Tensor(state.reshape(1, -1)).to(device)        
        probs = F.softmax(self.actor(Variable(state, volatile = True))*100)
        # if self.total_time_steps < 10_000 and self.total_time_steps > 500:
        #     # probs = torch.rand(size=(1, 3))
        #     # probs /= probs.sum()
        #     self.actor_learn()
        # else:
        #     probs = F.softmax(self.actor(state), dim=-1) 

        # rotation = int(torch.where(action < -4, -5, torch.where(action > 4, 5, 0)))        
        action = int(probs.multinomial(1).squeeze(0))
        # action = torch.argmax(probs)
        actions = probs.data.numpy().flatten()        
        # print(f'self.total_time_steps : {self.total_time_steps} , action : {action} , actions: {actions}, type(actions) : {type(actions)}')
        self.last_state = state.numpy().flatten()
        self.last_actions = actions
        # self.last_action = action.data.numpy().flatten()
        self.steps_since_last_train += 1
        self.total_time_steps += 1
        if self.total_time_steps % 10000 == 0:
            self.save_models()   
        
        # return rotation 
        # return int(action[0])
        return action

    def train(self):
        for it in range(self.train_iterations):
            # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(self.batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Step 5: From the next state s’, the Actor target plays the next action a’
            next_action = self.actor_target(next_state)

            # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            # print(f'next_action.shape : {next_action.shape}, next_state.shape : {next_state.shape}, state.shape : {state.shape}, action.shape : {action.shape}')            

            # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)

            # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
            target_Q = reward + ((1 - done) * self.discount * target_Q).detach()

            # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
            current_Q1, current_Q2 = self.critic(state, action)

            # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
            if it % self.policy_freq == 0:
                action_from_actor = self.actor(state)
                # print(f' Performing Gradient ascent - it : {it}, policy_freq : {policy_freq}, state : {state.shape}, action_from_actor: {action_from_actor.shape}')
                actor_loss = -self.critic.Q1(state, action_from_actor).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        print(f'critic_loss : {critic_loss} & actor_loss : {actor_loss}')

    def actor_learn(self):
        # print(f'actor_learn')
        self.batch_size = 50
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.replay_buffer.sample(self.batch_size)
        state = torch.Tensor(batch_states).to(device)
        next_state = torch.Tensor(batch_next_states).to(device)
        action = torch.Tensor(batch_actions).to(device)
        reward = torch.Tensor(batch_rewards).to(device)
        done = torch.Tensor(batch_dones).to(device)            
        
        outputs = self.actor(state)
        next_outputs = self.actor(next_state)
        target = self.gamma*next_outputs + reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.actor_optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.actor_optimizer.step()
        # print(f'actor_learn - td_loss : {td_loss}')

    # Making a save method to save models
    def save_models(self):
        torch.save(self.actor.state_dict(), 'last_actor.pth')
        torch.save(self.critic.state_dict(), 'last_critic.pth')
        print(f'Critic & actor models saved at {self.total_time_steps} steps')

    # Making a load method to load pre-trained models
    def load_models(self):
        if os.path.isfile('last_actor.pth') and os.path.isfile('last_critic.pth'):
            self.actor.load_state_dict(torch.load('last_actor.pth'))
            self.critic.load_state_dict(torch.load('last_critic.pth'))
            print(f'Critic & actor models loaded from checkpoints...')
        else:
            print("no checkpoint found...")
