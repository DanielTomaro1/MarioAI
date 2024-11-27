import torch
import numpy as np
from collections import deque
import random
from model import MarioNet

class Mario:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        # Remove the old epsilon_decay since we'll use a progressive decay
        # self.epsilon_decay = 0.99
        self.learning_rate = 0.00025
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.training_step = 0
        self.best_reward = float('-inf')

    def update_exploration_rate(self, episode):
        # Add this new method
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * (0.995 ** episode)  # Slower decay
        )

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        # Remove the channel dimension if it exists
        if len(state.shape) == 5:  # [batch, frames, channel, height, width]
            state = state.squeeze(2)  # Remove the channel dimension
        
        with torch.no_grad():
            action_values = self.net(state)
        action = torch.argmax(action_values, axis=1).item()
        
        return action 
       
    def cache(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, next_states, actions, rewards, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Reshape if using frame stacking
        if len(states.shape) == 5:  # [batch, frames, channel, height, width]
            states = states.squeeze(2)  # Remove channel dim as frames are now channels
            next_states = next_states.squeeze(2)
        
        current_q_values = self.net(states).gather(1, actions.unsqueeze(-1))
        
        with torch.no_grad():
            next_q_values = self.net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_step += 1
        return loss.item()
    
    def save_model(self, path, episode, total_reward):
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            torch.save({
                'episode': episode,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'best_reward': self.best_reward,
                'training_step': self.training_step
            }, path)
            return True
        return False
