# monitor.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
from collections import deque
import json

class MarioMonitor:
    def __init__(self, log_dir="logs", window_size=100):
        """
        Initialize the monitor with tracking metrics.
        
        Args:
            log_dir (str): Directory for saving logs and visualizations
            window_size (int): Size of the moving average window
        """
        self.log_dir = log_dir
        self.window_size = window_size
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tracking metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_kills = []
        self.episode_scores = []
        self.episode_blocks = []
        self.epsilons = []
        self.moving_avg_reward = deque(maxlen=window_size)
        
        # Performance tracking
        self.best_reward = float('-inf')
        self.best_score = 0
        self.total_kills = 0
        self.total_blocks = 0
        
        # Initialize the timestamp for this training session
        self.start_time = datetime.now()
        self.session_id = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Create session directory
        self.session_dir = os.path.join(log_dir, f"session_{self.session_id}")
        os.makedirs(self.session_dir, exist_ok=True)

    def update(self, stats):
        """
        Update monitor with latest episode statistics.
        
        Args:
            stats (dict): Dictionary containing episode statistics
        """
        # Extract stats
        episode = stats['episode']
        reward = stats['reward']
        kills = stats['kills']
        score = stats['score']
        blocks = stats['blocks_hit']
        epsilon = stats['epsilon']
        steps = stats['steps']
        
        # Update tracking lists
        self.episode_rewards.append(reward)
        self.episode_lengths.append(steps)
        self.episode_kills.append(kills)
        self.episode_scores.append(score)
        self.episode_blocks.append(blocks)
        self.epsilons.append(epsilon)
        
        # Update moving average
        self.moving_avg_reward.append(reward)
        
        # Update best performances
        self.best_reward = max(self.best_reward, reward)
        self.best_score = max(self.best_score, score)
        self.total_kills += kills
        self.total_blocks += blocks
        
        # Save episode data
        self._save_episode_data(stats)

    def plot_training_progress(self):
        """Generate and save training progress plots."""
        plt.figure(figsize=(15, 10))
        plt.clf()
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.episode_rewards, label='Episode Reward', alpha=0.6)
        plt.plot(pd.Series(self.episode_rewards).rolling(self.window_size).mean(), 
                label=f'{self.window_size}-Episode Moving Average', 
                linewidth=2)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        
        # Plot epsilon decay
        plt.subplot(2, 2, 2)
        plt.plot(self.epsilons, label='Epsilon', color='green')
        plt.title('Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon Value')
        plt.legend()
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(2, 2, 3)
        plt.plot(self.episode_lengths, label='Episode Length', color='orange')
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.legend()
        plt.grid(True)
        
        # Plot scores
        plt.subplot(2, 2, 4)
        plt.plot(self.episode_scores, label='Game Score', color='purple')
        plt.title('Game Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.session_dir, 'training_progress.png'))
        plt.close()

    def plot_performance_heatmap(self):
        """Generate and save performance heatmap."""
        if not self.episode_rewards:  # Check if we have any data
            print("No data available for heatmap yet")
            return
            
        # Create performance matrix
        episodes = len(self.episode_rewards)
        data = {
            'Rewards': pd.Series(self.episode_rewards).rolling(10).mean(),
            'Kills': pd.Series(self.episode_kills).rolling(10).mean(),
            'Scores': pd.Series(self.episode_scores).rolling(10).mean(),
            'Steps': pd.Series(self.episode_lengths).rolling(10).mean()
        }
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.clf()
        df = pd.DataFrame(data)
        
        # Normalize the data
        normalized_df = (df - df.min()) / (df.max() - df.min())
        
        if not normalized_df.empty:
            # Create heatmap
            sns.heatmap(normalized_df.T, 
                        cmap='viridis',
                        cbar_kws={'label': 'Normalized Performance'})
            
            plt.title('Performance Heatmap (10-Episode Rolling Average)')
            plt.xlabel('Episode')
            plt.tight_layout()
            plt.savefig(os.path.join(self.session_dir, 'performance_heatmap.png'))
        plt.close()

    def generate_stats_summary(self):
        """Generate a text summary of training statistics."""
        current_epsilon = self.epsilons[-1] if self.epsilons else 0.0
        
        summary = (
            f"\n=== Training Statistics ===\n"
            f"Total Episodes: {len(self.episode_rewards)}\n"
            f"Best Reward: {self.best_reward:.2f}\n"
            f"Best Score: {self.best_score}\n"
            f"Total Kills: {self.total_kills}\n"
            f"Total Blocks Hit: {self.total_blocks}\n"
        )

        if len(self.episode_rewards) >= 1:
            window_size = min(self.window_size, len(self.episode_rewards))
            avg_reward = np.mean(self.episode_rewards[-window_size:])
            avg_steps = np.mean(self.episode_lengths[-window_size:] if self.episode_lengths else [0])
            avg_score = np.mean(self.episode_scores[-window_size:] if self.episode_scores else [0])
            
            summary += (
                f"Last {window_size} Episodes:\n"
                f"  Average Reward: {avg_reward:.2f}\n"
                f"  Average Steps: {avg_steps:.2f}\n"
                f"  Average Score: {avg_score:.2f}\n"
            )
        
        summary += f"Current Epsilon: {current_epsilon:.3f}\n"
        summary += f"Training Duration: {datetime.now() - self.start_time}"
        
        # Save summary to file
        with open(os.path.join(self.session_dir, 'training_summary.txt'), 'w') as f:
            f.write(summary)
        
        return summary

    def _save_episode_data(self, stats):
        """Save episode data to JSON file."""
        stats['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        file_path = os.path.join(self.session_dir, 'episode_data.json')
        mode = 'a' if os.path.exists(file_path) else 'w'
        
        with open(file_path, mode) as f:
            json.dump(stats, f)
            f.write('\n')

    def plot_kill_distribution(self):
        """Generate and save kill distribution plot."""
        plt.figure(figsize=(10, 6))
        plt.clf()
        
        # Create kill distribution histogram
        plt.hist(self.episode_kills, bins=20, color='red', alpha=0.7)
        plt.title('Distribution of Enemies Killed per Episode')
        plt.xlabel('Number of Kills')
        plt.ylabel('Frequency')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.session_dir, 'kill_distribution.png'))
        plt.close()

    def plot_learning_efficiency(self):
        """Generate and save learning efficiency plot."""
        plt.figure(figsize=(10, 6))
        plt.clf()
        
        # Calculate reward per step
        efficiency = [r/s if s > 0 else 0 for r, s in zip(self.episode_rewards, self.episode_lengths)]
        
        plt.plot(efficiency, label='Reward per Step', color='blue', alpha=0.6)
        plt.plot(pd.Series(efficiency).rolling(self.window_size).mean(), 
                label=f'{self.window_size}-Episode Moving Average',
                color='red',
                linewidth=2)
        
        plt.title('Learning Efficiency (Reward per Step)')
        plt.xlabel('Episode')
        plt.ylabel('Efficiency')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.session_dir, 'learning_efficiency.png'))
        plt.close()

    def save_checkpoint(self):
        """Save monitor state to checkpoint."""
        checkpoint = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_kills': self.episode_kills,
            'episode_scores': self.episode_scores,
            'episode_blocks': self.episode_blocks,
            'epsilons': self.epsilons,
            'best_reward': self.best_reward,
            'best_score': self.best_score,
            'total_kills': self.total_kills,
            'total_blocks': self.total_blocks,
            'session_id': self.session_id,
            'start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        checkpoint_path = os.path.join(self.session_dir, 'monitor_checkpoint.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=4)

    def load_checkpoint(self, checkpoint_path):
        """Load monitor state from checkpoint."""
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.episode_kills = checkpoint['episode_kills']
        self.episode_scores = checkpoint['episode_scores']
        self.episode_blocks = checkpoint['episode_blocks']
        self.epsilons = checkpoint['epsilons']
        self.best_reward = checkpoint['best_reward']
        self.best_score = checkpoint['best_score']
        self.total_kills = checkpoint['total_kills']
        self.total_blocks = checkpoint['total_blocks']
        self.session_id = checkpoint['session_id']
        self.start_time = datetime.strptime(checkpoint['start_time'], "%Y-%m-%d %H:%M:%S")

    def get_current_performance_metrics(self):
        """Get current performance metrics as a dictionary."""
        if not self.episode_rewards:
            return {
                'current_episode': 0,
                'recent_avg_reward': 0,
                'recent_avg_steps': 0,
                'recent_avg_score': 0,
                'recent_avg_kills': 0,
                'best_reward': 0,
                'best_score': 0,
                'total_kills': 0,
                'current_epsilon': None,
                'training_duration': str(datetime.now() - self.start_time)
            }
                
        return {
            'current_episode': len(self.episode_rewards),
            'recent_avg_reward': np.mean(self.episode_rewards[-self.window_size:]),
            'recent_avg_steps': np.mean(self.episode_lengths[-self.window_size:]),
            'recent_avg_score': np.mean(self.episode_scores[-self.window_size:]),
            'recent_avg_kills': np.mean(self.episode_kills[-self.window_size:]),
            'best_reward': self.best_reward,
            'best_score': self.best_score,
            'total_kills': self.total_kills,
            'current_epsilon': self.epsilons[-1] if self.epsilons else None,
            'training_duration': str(datetime.now() - self.start_time)
        }