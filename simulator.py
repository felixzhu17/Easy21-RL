from game import Easy21
from tqdm import tqdm
import numpy as np
import torch

class MonteCarloSimulator:
    def __init__(self, num_episodes, policy, game_class=Easy21):
        self.num_episodes = num_episodes
        self.policy = policy
        self.game_class = game_class

    def run(self):
        for _ in tqdm(range(self.num_episodes)):
            game = self.game_class()
            while game.state.terminal is not True:
                action = self.policy.get_epsilon_greedy_action(
                    game.state.player_sum, game.state.dealer_sum
                )
                game.step(action)
            for _, row in game.get_history().iterrows():
                self.update(
                    player_sum=row["player_sum_t"],
                    dealer_sum=row["dealer_sum_t"],
                    action=row["action_t"],
                    cumulative_reward=row["cumulative_reward"],
                )

    def update(self, player_sum, dealer_sum, action, cumulative_reward):
        alpha = self.policy.get_update_alpha(player_sum, dealer_sum, action)
        self.policy.action_state_value[player_sum, dealer_sum, action] += alpha * (
            cumulative_reward
            - self.policy.get_action_state_value(player_sum, dealer_sum, action)
        )
        self.policy.action_state_counts[player_sum, dealer_sum, action] += 1


class SARSALambdaSimulator:
    def __init__(self, num_episodes, policy, lambda_, game_class=Easy21):
        self.num_episodes = num_episodes
        self.policy = policy
        self.lambda_ = lambda_
        self.game_class = game_class

    def run(self):
        for _ in tqdm(range(self.num_episodes)):
            game = self.game_class()
            eligibility = np.zeros_like(self.policy.action_state_value)
            action = self.policy.get_epsilon_greedy_action(
                game.state.player_sum, game.state.dealer_sum
            )
            previous_value = self.policy.get_action_state_value(
                game.state.player_sum, game.state.dealer_sum, action.value
            )

            while True:
                state_update = game.step(action)
                prev_player_sum, prev_dealer_sum, prev_action, reward = (
                    state_update.state_t.player_sum,
                    state_update.state_t.dealer_sum,
                    state_update.action_t.value,
                    state_update.reward_t_1,
                )
                if game.state.terminal:
                    expected_value = 0
                    sarsa_error = reward + expected_value - previous_value
                    eligibility = self.update(
                        prev_player_sum,
                        prev_dealer_sum,
                        prev_action,
                        sarsa_error,
                        eligibility,
                    )
                    break
                else:
                    action = self.policy.get_epsilon_greedy_action(
                        game.state.player_sum, game.state.dealer_sum
                    )
                    expected_value = self.policy.get_action_state_value(
                        game.state.player_sum, game.state.dealer_sum, action.value
                    )
                    sarsa_error = reward + expected_value - previous_value
                    eligibility = self.update(
                        prev_player_sum,
                        prev_dealer_sum,
                        prev_action,
                        sarsa_error,
                        eligibility,
                    )
                    previous_value = expected_value

    def update(self, player_sum, dealer_sum, action, sarsa_error, eligibility):
        eligibility *= self.lambda_ 
        eligibility[player_sum, dealer_sum, action] += 1
        alpha = self.policy.get_update_alpha(player_sum, dealer_sum, action)
        self.policy.action_state_value += alpha * sarsa_error * eligibility
        self.policy.action_state_counts[player_sum, dealer_sum, action] += 1
        return eligibility

class ApproximationSimulator:
    def __init__(self, num_episodes, model, lambda_, learning_rate, game_class=Easy21):
        self.num_episodes = num_episodes
        self.model = model
        self.game_class = game_class
        self.lambda_ = lambda_
        self.learning_rate = learning_rate

    def run(self):
        for _ in tqdm(range(self.num_episodes)):
            game = self.game_class()
            eligibility = {name: torch.zeros_like(param) 
                           for name, param in self.model.named_parameters()}
            action = self.model.get_epsilon_greedy_action(
                game.state.player_sum, game.state.dealer_sum
            )

            while True:
                state_update = game.step(action)
                prev_player_sum, prev_dealer_sum, prev_action, reward = (
                    state_update.state_t.player_sum,
                    state_update.state_t.dealer_sum,
                    state_update.action_t.value,
                    torch.tensor(state_update.reward_t_1),
                )
        
                if game.state.terminal:
                    expected_value = torch.tensor(0, requires_grad = False)
                    previous_value = self.model(prev_player_sum, prev_dealer_sum, prev_action)
                    sarsa_error = reward + expected_value - previous_value
                    eligibility = self.update(
                        sarsa_error,
                        eligibility,
                    )
                    break

                else:
                    action = self.model.get_epsilon_greedy_action(
                        game.state.player_sum, game.state.dealer_sum
                    )
                    with torch.no_grad():
                        expected_value = self.model(
                            game.state.player_sum, game.state.dealer_sum, action.value
                        )
                    previous_value = self.model(prev_player_sum, prev_dealer_sum, prev_action)
                    sarsa_error = reward + expected_value - previous_value
                    eligibility = self.update(
                        sarsa_error,
                        eligibility,
                    )

    def update(self, sarsa_error, eligibility):

        # MSE loss
        loss = (sarsa_error ** 2).mean()
        # Backward pass to compute gradients
        loss.backward()
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Discount and accumulate gradients in eligibility trace
                eligibility[name] *= self.lambda_
                eligibility[name] += param.grad
                
                # Update weights with eligibility trace
                param.data -= self.learning_rate * eligibility[name]

                # Reset gradients
                param.grad.zero_()

        return eligibility