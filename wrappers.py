# wrappers.py
import gym
import cv2
import numpy as np
from collections import deque

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

class CV2Renderer(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.window_name = "Mario AI Training"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)  # Larger window
        self.base_env = env
        
        # Try to get to the original env
        while hasattr(self.base_env, 'env'):
            self.base_env = self.base_env.env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Get original render
        try:
            screen = self.base_env.render(mode='rgb_array')
            if screen is not None:
                # Convert to BGR for OpenCV
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                
                # Upscale if too small
                if screen.shape[1] < 400:  # if width is less than 400
                    screen = cv2.resize(screen, (screen.shape[1]*2, screen.shape[0]*2), 
                                      interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow(self.window_name, screen)
                cv2.waitKey(1)
        except Exception as e:
            print(f"Render error: {e}")
        
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        
        # Get original render
        try:
            screen = self.base_env.render(mode='rgb_array')
            if screen is not None:
                screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                
                # Upscale if too small
                if screen.shape[1] < 400:
                    screen = cv2.resize(screen, (screen.shape[1]*2, screen.shape[0]*2), 
                                      interpolation=cv2.INTER_NEAREST)
                
                cv2.imshow(self.window_name, screen)
                cv2.waitKey(1)
        except Exception as e:
            print(f"Render error: {e}")
        
        return obs

    def close(self):
        cv2.destroyAllWindows()
        return self.env.close()