# main.py
import os
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import numpy as np
import time
from datetime import datetime, timedelta
import traceback
import gc
import torch
from config import CHECKPOINT_DIR, LOG_DIR, SIMPLE_MOVEMENT
from wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, CV2Renderer
from agent import Mario
from utils import display_status
from reward_handler import RewardHandler
from monitor import MarioMonitor

class MarioTrainer:
    def __init__(self):
        # Initialize environment
        self.env = self._setup_environment()
        
        # Initialize components
        self.state_dim = (1, 84, 84)
        self.action_dim = self.env.action_space.n
        self.mario = Mario(self.state_dim, self.action_dim)
        self.reward_handler = RewardHandler()
        self.monitor = MarioMonitor()
        
        # Training parameters
        self.running_jump_state = 0
        self.running_jump_start_pos = 0
        self.stuck_counter = 0
        self.start_time = time.time()
        self.total_episodes = 1000
        
        print(f"Using device: {self.mario.device}")

    def _setup_environment(self):
        try:
            env = gym_super_mario_bros.make('SuperMarioBros-v0')
            env = JoypadSpace(env, SIMPLE_MOVEMENT)
            env = SkipFrame(env)
            env = GrayScaleObservation(env)
            env = ResizeObservation(env)
            env = CV2Renderer(env)  # Use CV2 rendering
            return env
        except Exception as e:
            print(f"Error setting up environment: {str(e)}")
            raise

    def train(self, episodes=1000, max_steps=1000):
        print("Starting Mario AI Training...")
        start_time = time.time()
        self.total_episodes = episodes
        
        try:
            for episode in range(episodes):
                episode_stats = self._run_episode(episode, max_steps)
                self._process_episode_results(episode_stats, episode, start_time)
                
                # Cleanup between episodes
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self._handle_training_end("interrupted")
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            traceback.print_exc()
            self._handle_training_end("error")
        finally:
            self.env.close()

    def _run_episode(self, episode, max_steps):
        state = self.env.reset()
        state = np.array(state, dtype=np.float32) / 255.0
        
        # Reset episode tracking
        self.reward_handler = RewardHandler()
        self.running_jump_state = 0
        self.stuck_counter = 0
        
        total_reward = 0
        steps = 0
        current_loss = None
        info = {'x_pos': 0, 'y_pos': 0, 'score': 0, 'coins': 0, 'kills': 0}
        
        try:
            while steps < max_steps:
                steps += 1
                
                x_pos = prev_x_pos = self.reward_handler.prev_x_pos
                y_pos = prev_y_pos = self.reward_handler.prev_y_pos
                
                # Calculate y_pos_change for action determination
                y_pos_change = y_pos - prev_y_pos
                
                action = self._determine_action(state, x_pos, prev_x_pos, y_pos_change)
                
                next_state, reward, done, step_info = self.env.step(action)
                next_state = np.array(next_state, dtype=np.float32) / 255.0
                info.update(step_info)
                
                x_pos = info.get('x_pos', 0)
                y_pos = info.get('y_pos', 0)
                x_pos_change = x_pos - prev_x_pos
                y_pos_change = y_pos - prev_y_pos
                
                current_reward = self.reward_handler.calculate_reward(
                    info, x_pos, y_pos, x_pos_change, y_pos_change
                )
                
                self.mario.cache(state, next_state, action, current_reward, done)
                current_loss = self.mario.learn()
                
                state = next_state
                total_reward += current_reward
                
                if steps % 30 == 0:
                    self._display_training_status(episode, steps, total_reward, 
                                            current_reward, current_loss, info,
                                            self.start_time)
                
                if done:
                    break
                
                time.sleep(0.01)
        
        except Exception as e:
            print(f"Error during step: {e}")
            traceback.print_exc()
        
        return {
            'episode': episode + 1,
            'reward': total_reward,
            'kills': info.get('kills', 0),
            'score': info.get('score', 0),
            'blocks_hit': len(self.reward_handler.blocks_hit),
            'epsilon': self.mario.epsilon,
            'steps': steps,
            'loss': current_loss if current_loss is not None else 0.0
        }
    
    def _determine_action(self, state, x_pos, prev_x_pos, y_pos_change=0):
        # Check if we should wait for a power-up
        if self.reward_handler.should_wait_for_powerup():
            return 0  # NOOP - wait for power-up
            
        # Check if we should return to a block
        if self.reward_handler.should_return_to_block(x_pos):
            block_x = self.reward_handler.last_block_hit_pos[0]
            if x_pos < block_x - 5:
                return 1  # Move right
            elif x_pos > block_x + 5:
                return 6  # Move left
            return 5  # Jump to hit block again

        if self.running_jump_state == 0:  # Normal state
            action = self.mario.act(state)
            
            # Check if stuck
            x_pos_change = x_pos - prev_x_pos
            if abs(x_pos_change) < 1:
                self.stuck_counter += 1
                if self.stuck_counter > 20:
                    self.running_jump_state = 1
                    self.running_jump_start_pos = x_pos
                    self.stuck_counter = 0
            else:
                self.stuck_counter = 0
                
        elif self.running_jump_state == 1:  # Backing up
            action = 6  # Move left
            if x_pos <= self.running_jump_start_pos - 64:  # Increased backup distance
                self.running_jump_state = 2
                
        elif self.running_jump_state == 2:  # Running forward
            action = 2  # Move right with A pressed
            if x_pos >= self.running_jump_start_pos - 32:
                self.running_jump_state = 3
                
        elif self.running_jump_state == 3:  # Long jumping
            action = 4  # Right + A + B
            if y_pos_change > 0:
                self.running_jump_state = 0
                
        return action

    def _process_episode_results(self, stats, episode, start_time):
        # Update monitor
        self.monitor.update(stats)
        
        # Generate visualizations periodically
        if (episode + 1) % 50 == 0:
            self.monitor.plot_training_progress()
            self.monitor.plot_performance_heatmap()
            self.monitor.plot_kill_distribution()
            self.monitor.plot_learning_efficiency()
            print("\n" + self.monitor.generate_stats_summary())
            
        # Save model checkpoint
        if (episode + 1) % 50 == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f'mario_net_{episode+1}.pth')
            if self.mario.save_model(save_path, episode, stats['reward']):
                print(f"\nNew best model saved! Reward: {stats['reward']:.2f}")
                
        # Save monitor checkpoint
        self.monitor.save_checkpoint()

    def _display_training_status(self, episode, steps, total_reward, current_reward, 
                               current_loss, info, start_time):
        training_time = str(timedelta(seconds=int(time.time() - start_time)))
        status_info = {
            "Episode": f"{episode+1}/{self.total_episodes}",
            "Steps": steps,
            "Total Reward": f"{total_reward:.2f}",
            "Last Reward": f"{current_reward:.2f}",
            "Score": info.get('score', 0),
            "Status": info.get('status', 'small'),
            "Position": f"({info.get('x_pos', 0):.0f}, {info.get('y_pos', 0):.0f})",
            "Epsilon": f"{self.mario.epsilon:.3f}",
            "Loss": f"{current_loss:.4f}" if current_loss else "N/A",
            "Training Time": training_time,
            "Lives": info.get('life', 'N/A'),
            "World": f"{info.get('world', 'N/A')}-{info.get('stage', 'N/A')}"
        }
        display_status(status_info)

    def _handle_training_end(self, reason="completed"):
        try:
            # Generate final visualizations
            self.monitor.plot_training_progress()
            self.monitor.plot_performance_heatmap()
            self.monitor.plot_kill_distribution()
            self.monitor.plot_learning_efficiency()
            
            # Save final model
            final_save_path = os.path.join(CHECKPOINT_DIR, 'mario_net_final.pth')
            metrics = self.monitor.get_current_performance_metrics()
            self.mario.save_model(final_save_path, metrics['current_episode'], 
                                metrics.get('best_reward', 0))
            
            # Print final statistics
            print("\nFinal Training Statistics:")
            print(self.monitor.generate_stats_summary())
            
            if reason == "interrupted":
                print("\nTraining was interrupted by user. Progress has been saved.")
            elif reason == "error":
                print("\nTraining stopped due to an error. Progress has been saved.")
            else:
                print("\nTraining completed successfully!")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        trainer = MarioTrainer()
        trainer.train()
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()