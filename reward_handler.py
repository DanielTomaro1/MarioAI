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
        self.enemy_detected = False
        self.safe_distance_maintained = False
        self.successful_jumps = 0

    def calculate_reward(self, info, x_pos, y_pos, x_pos_change, y_pos_change):
        current_reward = 0
        current_status = info.get('status', 'small')

        # Stuck penalty - more aggressive
        if abs(x_pos_change) < 1:
            current_reward -= 5  # Penalize getting stuck
            if x_pos < 120:  # Extra penalty for getting stuck early
                current_reward -= 10

        # Forward progress reward 
        if x_pos > self.max_x_pos_reached:
            progress = (x_pos - self.max_x_pos_reached)
            current_reward += progress * 1.0  # Increased reward
            self.max_x_pos_reached = x_pos

        # Backward movement penalty
        if x_pos < self.prev_x_pos:
            current_reward -= abs(x_pos - self.prev_x_pos) * 0.8

        # Coin collection reward
        coins_collected = info.get('coins', 0) - self.prev_coins
        if coins_collected > 0:
            current_reward += coins_collected * 30

        # Block hitting reward
        current_block_pos = (int(x_pos/16), int((y_pos+16)/16))
        if info.get('score', 0) > self.prev_score and current_block_pos not in self.blocks_hit:
            self.blocks_hit.add(current_block_pos)
            current_reward += 40
            self.last_block_hit_pos = (x_pos, y_pos)
            self.waiting_for_powerup = True
            self.powerup_wait_time = 0

        # Power-up collection reward
        if current_status != self.prev_status:
            if self.prev_status == 'small' and current_status in ['tall', 'fireball']:
                current_reward += 150
                self.waiting_for_powerup = False

        # Enemy kill reward with positioning reward
        current_kills = info.get('kills', 0)
        kills_this_step = current_kills - self.prev_kills
        if kills_this_step > 0:
            # Base kill reward
            if x_pos < 120:  # First enemy kill bonus
                current_reward += 150
            else:
                current_reward += 80 * kills_this_step

            # Bonus for aerial kills
            if y_pos_change > 0:
                current_reward += 50

            # Kill streak bonus
            self.kill_streak += 1
            current_reward += self.kill_streak * 20
            self.last_enemy_kill_pos = x_pos

        # Reset kill streak if no kills this step
        else:
            self.kill_streak = 0

        # Death penalty
        if info.get('life', 2) < 2:
            if x_pos < 120:  # Early game death
                current_reward -= 250
            else:
                current_reward -= 150
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