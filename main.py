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
import gc

# Configuration
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Simplified action space
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
]

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = deque(maxlen=2)

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,) + obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.expand_dims(observation, axis=0)
        return observation

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape=84):
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1,) + self.shape, dtype=np.uint8)

    def observation(self, observation):
        if observation.shape[0] == 1:
            observation = cv2.resize(observation[0], self.shape, interpolation=cv2.INTER_AREA)
            observation = np.expand_dims(observation, axis=0)
        else:
            observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
            observation = np.expand_dims(observation, axis=0)
        return observation

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
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.00025
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.training_step = 0
        self.best_reward = float('-inf')

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.net(state)
        action = torch.argmax(action_values, axis=1).item()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
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
    env = GrayScaleObservation(env)
    env = ResizeObservation(env)
    
    state_dim = (1, 84, 84)
    action_dim = env.action_space.n
    mario = Mario(state_dim, action_dim)
    
    print(f"Using device: {mario.device}")
    
    # Track previous state information
    prev_coins = 0
    prev_score = 0
    prev_x_pos = 0
    prev_y_pos = 0
    prev_status = 'small'
    prev_kills = 0
    blocks_hit = set()
    blocks_passed = set()
    stuck_x_pos = 0
    stuck_counter = 0
    running_jump_state = 0
    running_jump_start_pos = 0
    last_enemy_kill_pos = 0
    kill_streak = 0
    
    episodes = 1000
    try:
        for e in range(episodes):
            state = env.reset()
            state = np.array(state, dtype=np.float32) / 255.0
            
            # Reset episode tracking variables
            prev_coins = 0
            prev_score = 0
            prev_x_pos = 0
            prev_y_pos = 0
            prev_status = 'small'
            prev_kills = 0
            blocks_hit.clear()
            blocks_passed.clear()
            stuck_counter = 0
            running_jump_state = 0
            kill_streak = 0
            last_enemy_kill_pos = 0
            
            done = False
            total_reward = 0
            steps = 0
            current_loss = None
            
            while not done and steps < 1000:
                steps += 1
                
                env.render()
                
                # Get current position and check for blocks
                x_pos = info.get('x_pos', 0) if 'info' in locals() else prev_x_pos
                y_pos = info.get('y_pos', 0) if 'info' in locals() else prev_y_pos
                
                # Check for blocks above
                blocks_above = False
                for y_offset in range(16, 64, 16):
                    check_pos = (int(x_pos/16), int((y_pos+y_offset)/16))
                    if check_pos not in blocks_hit and check_pos not in blocks_passed:
                        blocks_above = True
                        break
                
                # Determine action based on state
                if running_jump_state == 0:  # Normal state
                    if blocks_above:
                        if random.random() < 0.7:
                            action = random.choice([2, 3, 4])  # Jump actions
                        else:
                            action = mario.act(state)
                    else:
                        action = mario.act(state)
                        
                        # Check if stuck
                        x_pos_change = x_pos - prev_x_pos
                        if abs(x_pos_change) < 1:
                            stuck_counter += 1
                            if stuck_counter > 20:
                                running_jump_state = 1
                                running_jump_start_pos = x_pos
                                stuck_counter = 0
                        else:
                            stuck_counter = 0
                
                elif running_jump_state == 1:  # Backing up
                    action = 6  # Move left
                    if x_pos <= running_jump_start_pos - 48:
                        running_jump_state = 2
                
                elif running_jump_state == 2:  # Running forward
                    action = 1  # Move right
                    if x_pos >= running_jump_start_pos - 32:
                        running_jump_state = 3
                
                elif running_jump_state == 3:  # Jumping
                    action = 4  # Right + A + B
                    if y_pos_change > 0:
                        running_jump_state = 0
                
                try:
                    next_state, reward, done, info = env.step(action)
                    next_state = np.array(next_state, dtype=np.float32) / 255.0
                    
                    current_reward = 0
                    
                    # Position tracking
                    x_pos = info.get('x_pos', 0)
                    y_pos = info.get('y_pos', 0)
                    x_pos_change = x_pos - prev_x_pos
                    y_pos_change = y_pos - prev_y_pos
                    current_status = info.get('status', 'small')
                    
                    # Enemy kill detection and rewards
                    current_kills = info.get('kills', 0)
                    kills_this_step = current_kills - prev_kills
                    
                    if kills_this_step > 0:
                        # Base kill reward
                        current_reward += 30 * kills_this_step  # Big base reward for kills
                        
                        # Kill streak bonus
                        kill_streak += kills_this_step
                        if kill_streak > 1:
                            current_reward += 15 * kill_streak  # Bonus for kill streaks
                        
                        # Style points for jumping kills
                        if y_pos > 50:  # If killing while in the air
                            current_reward += 20  # Extra reward for aerial kills
                        
                        # Multi-kill bonus
                        if kills_this_step > 1:
                            current_reward += 25 * kills_this_step  # Bonus for multiple kills at once
                        
                        last_enemy_kill_pos = x_pos
                    else:
                        kill_streak = 0
                    
                    # Block and coin rewards (maintained but reduced)
                    score_increase = info.get('score', 0) - prev_score
                    current_block_pos = (int(x_pos/16), int((y_pos+16)/16))
                    
                    if score_increase > 0 and current_block_pos not in blocks_hit:
                        blocks_hit.add(current_block_pos)
                        current_reward += 10  # Reduced from 20
                        if prev_status == 'small' and current_status == 'tall':
                            current_reward += 40  # Maintained high because it helps kill enemies
                    
                    # Running jump rewards
                    if running_jump_state > 0:
                        if running_jump_state == 3 and y_pos_change > 0:
                            current_reward += y_pos_change * 2
                        if x_pos > running_jump_start_pos:
                            current_reward += 15
                            running_jump_state = 0
                    
                    # Movement rewards
                    if x_pos_change > 0 and running_jump_state == 0:
                        current_reward += x_pos_change * 0.1
                    
                    # Coin collection
                    coins_collected = info.get('coins', 0) - prev_coins
                    if coins_collected > 0:
                        current_reward += coins_collected * 5  # Reduced from 8
                    
                    # Jumping rewards
                    if y_pos_change > 0:  # Moving upward
                        current_reward += y_pos_change * 0.3  # Encourage jumping for attack
                    
                    # Status-based rewards
                    if current_status == 'tall':
                        current_reward += 0.1  # Small bonus for maintaining powered-up state
                    
                    # Death penalty
                    if info.get('life', 2) < 2:
                        current_reward -= 50
                        running_jump_state = 0
                        kill_streak = 0
                    
                    # Update previous state tracking
                    prev_status = current_status
                    prev_coins = info.get('coins', 0)
                    prev_score = info.get('score', 0)
                    prev_x_pos = x_pos
                    prev_y_pos = y_pos
                    prev_kills = current_kills
                    
                    # Store and learn
                    mario.cache(state, next_state, action, current_reward, done)
                    current_loss = mario.learn()
                    
                    state = next_state
                    total_reward += current_reward
                    
                    # Display status every 30 steps
                    if steps % 30 == 0:
                        training_time = str(timedelta(seconds=int(time.time() - start_time)))
                        status_info = {
                            "Episode": f"{e+1}/{episodes}",
                            "Steps": steps,
                            "Total Reward": f"{total_reward:.2f}",
                            "Last Reward": f"{current_reward:.2f}",
                            "Kills": current_kills,
                            "Kill Streak": kill_streak,
                            "Score": info.get('score', 0),
                            "Status": current_status,
                            "Position": f"({x_pos:.0f}, {y_pos:.0f})",
                            "Blocks Hit": len(blocks_hit),
                            "Jump State": running_jump_state,
                            "Epsilon": f"{mario.epsilon:.3f}",
                            "Loss": f"{current_loss:.4f}" if current_loss else "N/A",
                            "Training Time": training_time,
                            "Lives": info.get('life', 'N/A'),
                            "World": f"{info.get('world', 'N/A')}-{info.get('stage', 'N/A')}"
                        }
                        display_status(status_info)
                    
                    time.sleep(0.01)
                
                except Exception as e:
                    print(f"Error during step: {e}")
                    traceback.print_exc()
                    break
            
            # End of episode summary
            print(f"\nEpisode {e+1} finished:")
            print(f"Total Steps: {steps}")
            print(f"Final Reward: {total_reward:.2f}")
            print(f"Total Kills: {current_kills}")
            print(f"Blocks Hit: {len(blocks_hit)}")
            print(f"Final Score: {prev_score}")
            print(f"Coins Collected: {prev_coins}")
            
            # Save model periodically
            if (e + 1) % 50 == 0:
                save_path = os.path.join(CHECKPOINT_DIR, f'mario_net_{e+1}.pth')
                if mario.save_model(save_path, e, total_reward):
                    print(f"\nNew best model saved! Reward: {total_reward:.2f}")
            
            # Log episode stats
            with open(os.path.join(LOG_DIR, 'training_log.txt'), 'a') as f:
                f.write(f"Episode {e+1}: Steps={steps}, Reward={total_reward:.2f}, " 
                        f"Kills={current_kills}, Score={prev_score}, "
                        f"BlocksHit={len(blocks_hit)}, Epsilon={mario.epsilon:.3f}\n")
            
            # Cleanup between episodes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
