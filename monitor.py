# monitor.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class MarioMonitor:
    def __init__(self, log_dir="logs", checkpoint_dir="checkpoints"):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.stats = {
            'episodes': [],
            'rewards': [],
            'kills': [],
            'scores': [],
            'blocks_hit': [],
            'epsilon': [],
            'steps': [],
            'avg_reward_per_step': []
        }
        self.rolling_window = 50  # For moving averages
        
    def update(self, episode_stats):
        """Update stats with new episode data"""
        self.stats['episodes'].append(episode_stats['episode'])
        self.stats['rewards'].append(episode_stats['reward'])
        self.stats['kills'].append(episode_stats['kills'])
        self.stats['scores'].append(episode_stats['score'])
        self.stats['blocks_hit'].append(episode_stats['blocks_hit'])
        self.stats['epsilon'].append(episode_stats['epsilon'])
        self.stats['steps'].append(episode_stats['steps'])
        self.stats['avg_reward_per_step'].append(episode_stats['reward'] / max(1, episode_stats['steps']))
        
        # Save stats to file
        self.save_stats()
        
    def save_stats(self):
        """Save current stats to JSON file"""
        stats_file = os.path.join(self.log_dir, 'training_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f)
            
    def load_stats(self):
        """Load stats from JSON file"""
        stats_file = os.path.join(self.log_dir, 'training_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
                
    def plot_training_progress(self):
        """Generate comprehensive training progress plots"""
        plt.style.use('seaborn')
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Mario AI Training Progress', size=16)
        
        # Calculate moving averages
        def moving_average(data, window=self.rolling_window):
            return pd.Series(data).rolling(window=window, min_periods=1).mean()
        
        # Plot 1: Rewards over time
        ax = axes[0, 0]
        episodes = self.stats['episodes']
        rewards = self.stats['rewards']
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        ax.plot(episodes, moving_average(rewards), color='blue', label=f'{self.rolling_window}-Episode Average')
        ax.set_title('Rewards per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        
        # Plot 2: Kills over time
        ax = axes[0, 1]
        kills = self.stats['kills']
        ax.plot(episodes, kills, alpha=0.3, color='red', label='Raw')
        ax.plot(episodes, moving_average(kills), color='red', label=f'{self.rolling_window}-Episode Average')
        ax.set_title('Enemies Killed per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Kills')
        ax.legend()
        
        # Plot 3: Blocks hit over time
        ax = axes[1, 0]
        blocks = self.stats['blocks_hit']
        ax.plot(episodes, blocks, alpha=0.3, color='green', label='Raw')
        ax.plot(episodes, moving_average(blocks), color='green', label=f'{self.rolling_window}-Episode Average')
        ax.set_title('Blocks Hit per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Blocks')
        ax.legend()
        
        # Plot 4: Scores over time
        ax = axes[1, 1]
        scores = self.stats['scores']
        ax.plot(episodes, scores, alpha=0.3, color='purple', label='Raw')
        ax.plot(episodes, moving_average(scores), color='purple', label=f'{self.rolling_window}-Episode Average')
        ax.set_title('Game Score per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.legend()
        
        # Plot 5: Steps per episode
        ax = axes[2, 0]
        steps = self.stats['steps']
        ax.plot(episodes, steps, alpha=0.3, color='orange', label='Raw')
        ax.plot(episodes, moving_average(steps), color='orange', label=f'{self.rolling_window}-Episode Average')
        ax.set_title('Steps per Episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.legend()
        
        # Plot 6: Epsilon decay
        ax = axes[2, 1]
        epsilon = self.stats['epsilon']
        ax.plot(episodes, epsilon, color='brown')
        ax.set_title('Exploration Rate (Epsilon)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(self.log_dir, 'training_progress.png'))
        plt.close()
        
    def plot_performance_heatmap(self):
        """Generate a heatmap showing relationships between different metrics"""
        metrics = ['rewards', 'kills', 'blocks_hit', 'scores', 'steps']
        data = np.zeros((len(metrics), len(metrics)))
        
        # Calculate correlations
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                corr = np.corrcoef(self.stats[m1], self.stats[m2])[0, 1]
                data[i, j] = corr
                
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
        
        # Add labels
        plt.xticks(range(len(metrics)), metrics, rotation=45)
        plt.yticks(range(len(metrics)), metrics)
        
        # Add colorbar
        plt.colorbar(label='Correlation')
        
        # Add correlation values
        for i in range(len(metrics)):
            for j in range(len(metrics)):
                plt.text(j, i, f'{data[i, j]:.2f}', 
                        ha='center', va='center')
                
        plt.title('Performance Metrics Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'correlation_heatmap.png'))
        plt.close()

    def generate_stats_summary(self):
        """Generate statistical summary of training progress"""
        recent_episodes = 100  # Number of recent episodes to compare against
        
        if len(self.stats['episodes']) == 0:
            return "No training data available yet."
            
        recent_data = {
            'rewards': self.stats['rewards'][-recent_episodes:],
            'kills': self.stats['kills'][-recent_episodes:],
            'blocks_hit': self.stats['blocks_hit'][-recent_episodes:],
            'scores': self.stats['scores'][-recent_episodes:],
            'steps': self.stats['steps'][-recent_episodes:]
        }
        
        all_time_best = {
            'reward': max(self.stats['rewards']),
            'kills': max(self.stats['kills']),
            'blocks_hit': max(self.stats['blocks_hit']),
            'score': max(self.stats['scores']),
            'steps': max(self.stats['steps'])
        }
        
        summary = (
            f"\nTraining Statistics Summary:\n"
            f"==========================\n"
            f"Total Episodes: {len(self.stats['episodes'])}\n"
            f"\nRecent Performance (Last {recent_episodes} episodes):\n"
            f"Average Reward: {np.mean(recent_data['rewards']):.2f}\n"
            f"Average Kills: {np.mean(recent_data['kills']):.2f}\n"
            f"Average Blocks Hit: {np.mean(recent_data['blocks_hit']):.2f}\n"
            f"Average Score: {np.mean(recent_data['scores']):.2f}\n"
            f"Average Steps: {np.mean(recent_data['steps']):.2f}\n"
            f"\nAll-Time Best Performance:\n"
            f"Best Reward: {all_time_best['reward']:.2f}\n"
            f"Most Kills: {all_time_best['kills']}\n"
            f"Most Blocks Hit: {all_time_best['blocks_hit']}\n"
            f"Highest Score: {all_time_best['score']}\n"
            f"Longest Episode: {all_time_best['steps']} steps\n"
            f"\nCurrent Exploration Rate: {self.stats['epsilon'][-1]:.3f}\n"
        )
        
        # Save summary to file
        with open(os.path.join(self.log_dir, 'stats_summary.txt'), 'w') as f:
            f.write(summary)
            
        return summary