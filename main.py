# main.py
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import cv2
import os
import time
from datetime import datetime, timedelta
import traceback

# Configuration
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, input):
        return self.online(input)

class Mario:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99995
        self.learning_rate = 0.00025
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.training_step = 0
        self.best_reward = float('-inf')

    def act(self, state):
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)
        
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).to(self.device)
        action_values = self.net(state)
        action = torch.argmax(action_values, axis=1).item()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action

    def cache(self, state, next_state, action, reward, done):
        if len(state.shape) == 3:
            state = np.expand_dims(state, axis=0)
        if len(next_state.shape) == 3:
            next_state = np.expand_dims(next_state, axis=0)
        
        self.memory.append((state, next_state, action, reward, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        
        batch = random.sample(self.memory, self.batch_size)
        states, next_states, actions, rewards, dones = zip(*batch)
        
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.net(states).gather(1, actions.unsqueeze(-1))
        
        with torch.no_grad():
            next_q_values = self.net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
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

def preprocess_observation(observation):
    try:
        if len(observation.shape) == 2:
            observation = np.expand_dims(observation, axis=0)
        elif len(observation.shape) == 3:
            if observation.shape[-1] == 3:
                observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
                observation = np.expand_dims(observation, axis=0)
        
        if observation.shape[1:] != (84, 84):
            observation = cv2.resize(observation[0], (84, 84), interpolation=cv2.INTER_AREA)
            observation = np.expand_dims(observation, axis=0)
        
        return observation.astype(np.float32) / 255.0
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        traceback.print_exc()
        return observation

def display_status(info_dict):
    status = "\n=== Mario AI Training Status ===\n"
    for key, value in info_dict.items():
        status += f"{key}: {value}\n"
    status += "\nPress Ctrl+C to save and exit\n"
    print(status)

def train():
    print("Starting Mario AI Training...")
    start_time = time.time()
    
    # Initialize environment
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env)
    
    state_dim = (1, 84, 84)
    action_dim = env.action_space.n
    mario = Mario(state_dim, action_dim)
    
    print(f"Using device: {mario.device}")
    
    episodes = 1000
    try:
        for e in range(episodes):
            state = env.reset()
            state = preprocess_observation(state)
            
            done = False
            total_reward = 0
            steps = 0
            current_loss = None
            
            while not done and steps < 1000:
                steps += 1
                
                # Get action
                action = mario.act(state)
                
                # Perform action
                next_state, reward, done, info = env.step(action)
                next_state = preprocess_observation(next_state)
                
                # Enhanced reward shaping
                reward = reward + info.get('x_pos_change', 0) * 0.1
                if info.get('flag_get', False):
                    reward += 100
                if info.get('life', 2) < 2:
                    reward -= 50
                
                # Store and learn
                mario.cache(state, next_state, action, reward, done)
                current_loss = mario.learn()
                
                state = next_state
                total_reward += reward
                
                # Display status every 10 steps
                if steps % 10 == 0:
                    training_time = str(timedelta(seconds=int(time.time() - start_time)))
                    status_info = {
                        "Episode": f"{e+1}/{episodes}",
                        "Steps": steps,
                        "Reward": f"{total_reward:.2f}",
                        "Epsilon": f"{mario.epsilon:.3f}",
                        "Loss": f"{current_loss:.4f}" if current_loss else "N/A",
                        "Training Time": training_time
                    }
                    display_status(status_info)
                
                # Render every 20 episodes
                if e % 20 == 0:
                    env.render()
            
            # Save model periodically
            if (e + 1) % 50 == 0:
                save_path = os.path.join(CHECKPOINT_DIR, f'mario_net_{e+1}.pth')
                if mario.save_model(save_path, e, total_reward):
                    print(f"\nNew best model saved! Reward: {total_reward:.2f}")
            
            # Log episode stats
            with open(os.path.join(LOG_DIR, 'training_log.txt'), 'a') as f:
                f.write(f"Episode {e+1}: Steps={steps}, Reward={total_reward:.2f}, Epsilon={mario.epsilon:.3f}\n")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        traceback.print_exc()
    finally:
        # Save final model
        final_save_path = os.path.join(CHECKPOINT_DIR, 'mario_net_final.pth')
        mario.save_model(final_save_path, e, total_reward)
        
        # Close environment
        env.close()
        
        # Print summary
        total_time = str(timedelta(seconds=int(time.time() - start_time)))
        print("\nTraining Summary:")
        print(f"Total Episodes Completed: {e+1}")
        print(f"Best Reward Achieved: {mario.best_reward:.2f}")
        print(f"Final Epsilon: {mario.epsilon:.3f}")
        print(f"Total Training Time: {total_time}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()