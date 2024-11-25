# test_mario.py
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import time

def test_environment():
    try:
        # Create environment
        print("Creating environment...")
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        # Test environment
        print("Testing reset...")
        state = env.reset()
        print(f"State shape: {state.shape}")
        
        print("\nTesting actions...")
        for step in range(100):
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.05)
            
            if step % 10 == 0:
                print(f"Step {step}, Reward: {reward}")
            
            if done:
                state = env.reset()
        
        env.close()
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Starting Mario environment test...")
    test_environment()