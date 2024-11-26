# reward_handler.py
class RewardHandler:
    def __init__(self):
        self.prev_coins = 0
        self.prev_score = 0
        self.prev_x_pos = 0
        self.prev_y_pos = 0
        self.prev_status = 'small'
        self.prev_kills = 0
        self.blocks_hit = set()
        self.kill_streak = 0
        self.last_enemy_kill_pos = 0
        self.waiting_for_powerup = False
        self.powerup_wait_time = 0
        self.last_block_hit_pos = None
        self.block_contents_waiting_time = 0
        self.max_x_pos_reached = 0

    def calculate_reward(self, info, x_pos, y_pos, x_pos_change, y_pos_change):
        current_reward = 0
        current_status = info.get('status', 'small')

        # Forward progress reward
        if x_pos > self.max_x_pos_reached:
            current_reward += (x_pos - self.max_x_pos_reached) * 0.5
            self.max_x_pos_reached = x_pos

        # Coin collection reward
        coins_collected = info.get('coins', 0) - self.prev_coins
        if coins_collected > 0:
            current_reward += coins_collected * 20  # Significant reward for coins

        # Block hitting reward
        current_block_pos = (int(x_pos/16), int((y_pos+16)/16))
        if info.get('score', 0) > self.prev_score and current_block_pos not in self.blocks_hit:
            self.blocks_hit.add(current_block_pos)
            current_reward += 30  # Base reward for hitting block
            self.last_block_hit_pos = (x_pos, y_pos)
            self.waiting_for_powerup = True
            self.powerup_wait_time = 0

        # Power-up collection reward
        if current_status != self.prev_status:
            if self.prev_status == 'small' and current_status in ['tall', 'fireball']:
                current_reward += 100  # Big reward for getting power-up
                self.waiting_for_powerup = False

        # Enemy kill reward
        current_kills = info.get('kills', 0)
        kills_this_step = current_kills - self.prev_kills
        if kills_this_step > 0:
            current_reward += 50 * kills_this_step  # Base kill reward
            
            # Bonus for aerial kills
            if y_pos_change > 0:  # If killed while jumping
                current_reward += 30  # Extra reward for stylish kills
            
            self.kill_streak += 1
            current_reward += self.kill_streak * 10  # Bonus for kill streaks
            self.last_enemy_kill_pos = x_pos
        else:
            self.kill_streak = 0

        # Waiting for power-up behavior
        if self.waiting_for_powerup:
            self.powerup_wait_time += 1
            if self.powerup_wait_time < 30:  # Wait for about 1 second
                current_reward += 1  # Small reward for waiting
            else:
                self.waiting_for_powerup = False

        # Jumping reward
        if y_pos_change > 0:  # Moving upward
            current_reward += y_pos_change * 0.2  # Small reward for jumping

        # Death penalty
        if info.get('life', 2) < 2:
            current_reward -= 100
            self.kill_streak = 0

        # Update previous states
        self.prev_status = current_status
        self.prev_coins = info.get('coins', 0)
        self.prev_score = info.get('score', 0)
        self.prev_x_pos = x_pos
        self.prev_y_pos = y_pos
        self.prev_kills = current_kills

        return current_reward

    def should_wait_for_powerup(self):
        return self.waiting_for_powerup and self.powerup_wait_time < 30

    def should_return_to_block(self, current_x_pos):
        if self.last_block_hit_pos is None:
            return False
        distance_to_block = abs(current_x_pos - self.last_block_hit_pos[0])
        return distance_to_block < 48  # Return if block is nearby