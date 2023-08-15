from game import Easy21
from tqdm import tqdm
import numpy as np
import torch


class Evaluator:
    def __init__(self, policy, game_class=Easy21):
        self.policy = policy
        self.game_class = game_class

    def evaluate(self, num_episodes):
        score = 0
        for _ in tqdm(range(num_episodes)):
            game = self.game_class()
            while game.state.terminal is not True:
                action = self.policy.get_optimal_action(
                    game.state.player_sum, game.state.dealer_sum
                )
                game.step(action)
            score += game.history[-1].reward_t_1
        return score / num_episodes


class MonteCarloSimulator:
    def __init__(self, policy, game_class=Easy21):
        self.policy = policy
        self.game_class = game_class

    def run(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
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
    def __init__(self, policy, lambda_, game_class=Easy21):
        self.policy = policy
        self.lambda_ = lambda_
        self.game_class = game_class

    def run(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
            game = self.game_class()
            eligibility = np.zeros_like(self.policy.action_state_value)
            action = self.policy.get_epsilon_greedy_action(
                game.state.player_sum, game.state.dealer_sum
            )
            previous_expected_value = self.policy.get_action_state_value(
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
                    sarsa_error = reward + expected_value - previous_expected_value
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
                    sarsa_error = reward + expected_value - previous_expected_value
                    eligibility = self.update(
                        prev_player_sum,
                        prev_dealer_sum,
                        prev_action,
                        sarsa_error,
                        eligibility,
                    )
                    previous_expected_value = expected_value

    def update(self, player_sum, dealer_sum, action, sarsa_error, eligibility):
        eligibility *= self.lambda_
        eligibility[player_sum, dealer_sum, action] += 1
        alpha = self.policy.get_update_alpha(player_sum, dealer_sum, action)
        self.policy.action_state_value += alpha * sarsa_error * eligibility
        self.policy.action_state_counts[player_sum, dealer_sum, action] += 1
        return eligibility


class ApproximationSimulator:
    def __init__(self, model, lambda_, learning_rate, game_class=Easy21):
        self.model = model
        self.game_class = game_class
        self.lambda_ = lambda_
        self.learning_rate = learning_rate

    def run(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
            game = self.game_class()
            eligibility = self.init_eligibility()
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
                previous_expected_value = self.model(
                    prev_player_sum, prev_dealer_sum, prev_action
                )
                if game.state.terminal:
                    expected_value = torch.tensor(0, requires_grad=False)
                    sarsa_error = reward + expected_value - previous_expected_value
                    eligibility = self.update(
                        previous_expected_value,
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
                    sarsa_error = reward + expected_value - previous_expected_value
                    eligibility = self.update(
                        previous_expected_value,
                        sarsa_error,
                        eligibility,
                    )

    def update(self, estimate, sarsa_error, eligibility):

        # Compute gradient of the estimate
        estimate.backward()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                # Discount and accumulate gradients in eligibility trace
                eligibility[name] *= self.lambda_
                eligibility[name] += param.grad

                # Update weights with eligibility trace
                param.data += self.learning_rate * eligibility[name] * sarsa_error

                # Reset gradients
                param.grad.zero_()

        return eligibility

    def init_eligibility(self):
        return {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
        }


class REINFORCESimulator:
    def __init__(self, model, learning_rate, game_class=Easy21):
        self.model = model
        self.game_class = game_class
        self.learning_rate = learning_rate

    def run(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
            game = self.game_class()
            while game.state.terminal is not True:
                action = self.model.get_epsilon_greedy_action(
                    game.state.player_sum, game.state.dealer_sum
                )
                game.step(action)
            self.update(game.get_history())

    def update(self, history):
        loss = torch.tensor(0).float()
        for _, row in history.iterrows():
            player_sum, dealer_sum, action, reward = (
                row["player_sum_t"],
                row["dealer_sum_t"],
                row["action_t"],
                row["cumulative_reward"],
            )
            log_prob = torch.log(self.model(player_sum, dealer_sum)[action])
            loss += -log_prob * torch.tensor(reward).float()
        loss.backward()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data -= self.learning_rate * param.grad
                param.grad.zero_()


class ActorCriticSimulator:
    def __init__(
        self,
        actor_model,
        critic_model,
        lambda_,
        actor_learning_rate,
        critic_learning_rate,
        game_class=Easy21,
    ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.game_class = game_class
        self.lambda_ = lambda_
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

    def run(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
            game = self.game_class()
            eligibility = self.init_eligibility()
            action = self.actor_model.get_epsilon_greedy_action(
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
                with torch.no_grad():
                    previous_expected_value = self.critic_model(
                        prev_player_sum, prev_dealer_sum, prev_action
                    )
                self.update_actor(
                    prev_player_sum,
                    prev_dealer_sum,
                    prev_action,
                    previous_expected_value,
                )
                previous_expected_value = self.critic_model(
                    prev_player_sum, prev_dealer_sum, prev_action
                )
                if game.state.terminal:
                    expected_value = torch.tensor(0, requires_grad=False)
                    sarsa_error = reward + expected_value - previous_expected_value
                    eligibility = self.update_critic(
                        previous_expected_value, sarsa_error, eligibility
                    )
                    break

                else:
                    action = self.actor_model.get_epsilon_greedy_action(
                        game.state.player_sum, game.state.dealer_sum
                    )
                    with torch.no_grad():
                        expected_value = self.critic_model(
                            game.state.player_sum, game.state.dealer_sum, action.value
                        )
                    sarsa_error = reward + expected_value - previous_expected_value
                    eligibility = self.update_critic(
                        previous_expected_value, sarsa_error, eligibility
                    )

    def update_actor(self, player_sum, dealer_sum, action, reward):
        # REINFORCE loss
        log_prob = torch.log(self.actor_model(player_sum, dealer_sum)[action])
        loss = -log_prob * reward
        # Backward pass to compute gradients
        loss.backward()
        with torch.no_grad():
            for name, param in self.actor_model.named_parameters():
                param.data -= self.actor_learning_rate * param.grad
                # Reset gradients
                param.grad.zero_()

        return

    def update_critic(self, estimate, sarsa_error, eligibility):

        # Compute gradient of the estimate
        estimate.backward()

        with torch.no_grad():
            for name, param in self.critic_model.named_parameters():
                # Discount and accumulate gradients in eligibility trace
                eligibility[name] *= self.lambda_
                eligibility[name] += param.grad

                # Update weights with eligibility trace
                param.data += (
                    self.critic_learning_rate * eligibility[name] * sarsa_error
                )

                # Reset gradients
                param.grad.zero_()

        return eligibility

    def init_eligibility(self):
        return {
            name: torch.zeros_like(param)
            for name, param in self.critic_model.named_parameters()
        }


class AdvantageSimulator:
    def __init__(
        self,
        actor_model,
        critic_model,
        lambda_,
        actor_learning_rate,
        critic_learning_rate,
        game_class=Easy21,
    ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.game_class = game_class
        self.lambda_ = lambda_
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

    def run(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
            game = self.game_class()
            eligibility = self.init_eligibility()

            while True:
                action = self.actor_model.get_epsilon_greedy_action(
                    game.state.player_sum, game.state.dealer_sum
                )
                state_update = game.step(action)
                prev_player_sum, prev_dealer_sum, prev_action, reward = (
                    state_update.state_t.player_sum,
                    state_update.state_t.dealer_sum,
                    state_update.action_t.value,
                    torch.tensor(state_update.reward_t_1),
                )
                with torch.no_grad():
                    previous_expected_value = self.critic_model(
                        prev_player_sum, prev_dealer_sum
                    )

                if game.state.terminal:
                    expected_value = torch.tensor(0, requires_grad=False)
                    sarsa_error = reward + expected_value - previous_expected_value
                    self.update_actor(
                        prev_player_sum,
                        prev_dealer_sum,
                        prev_action,
                        sarsa_error,
                    )
                    previous_expected_value = self.critic_model(
                        prev_player_sum, prev_dealer_sum
                    )
                    sarsa_error = reward + expected_value - previous_expected_value
                    eligibility = self.update_critic(
                        previous_expected_value, sarsa_error, eligibility
                    )
                    break

                else:
                    with torch.no_grad():
                        expected_value = self.critic_model(
                            game.state.player_sum, game.state.dealer_sum
                        )
                    sarsa_error = reward + expected_value - previous_expected_value
                    self.update_actor(
                        prev_player_sum,
                        prev_dealer_sum,
                        prev_action,
                        sarsa_error,
                    )
                    previous_expected_value = self.critic_model(
                        prev_player_sum, prev_dealer_sum
                    )
                    eligibility = self.update_critic(
                        previous_expected_value, sarsa_error, eligibility
                    )

    def update_actor(self, player_sum, dealer_sum, action, reward):
        # REINFORCE loss
        log_prob = torch.log(self.actor_model(player_sum, dealer_sum)[action])
        loss = -log_prob * reward
        # Backward pass to compute gradients
        loss.backward()
        with torch.no_grad():
            for name, param in self.actor_model.named_parameters():
                param.data -= self.actor_learning_rate * param.grad
                # Reset gradients
                param.grad.zero_()

        return

    def update_critic(self, estimate, sarsa_error, eligibility):

        # Compute gradient of the estimate
        estimate.backward()

        with torch.no_grad():
            for name, param in self.critic_model.named_parameters():
                # Discount and accumulate gradients in eligibility trace
                eligibility[name] *= self.lambda_
                eligibility[name] += param.grad

                # Update weights with eligibility trace
                param.data += (
                    self.critic_learning_rate * eligibility[name] * sarsa_error
                )

                # Reset gradients
                param.grad.zero_()

        return eligibility

    def init_eligibility(self):
        return {
            name: torch.zeros_like(param)
            for name, param in self.critic_model.named_parameters()
        }


class PPOSimulator:
    def __init__(
        self,
        actor_model,
        critic_model,
        lambda_,
        actor_learning_rate,
        critic_learning_rate,
        epsilon,
        actor_iterations,
        critic_iterations,
        game_class=Easy21,
    ):
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.game_class = game_class
        self.lambda_ = lambda_
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.epsilon = epsilon
        self.actor_iterations = actor_iterations
        self.critic_iterations = critic_iterations

    def run(self, num_episodes):
        for _ in tqdm(range(num_episodes)):
            game = self.game_class()
            while game.state.terminal is not True:
                action, action_probability = self.actor_model.get_epsilon_greedy_action(
                    game.state.player_sum,
                    game.state.dealer_sum,
                    return_probability=True,
                )
                game.step(action, action_probability)
            game_history = game.get_history()

            for _ in range(self.critic_iterations):
                self.update_critic(game_history)

            for _ in range(self.actor_iterations):
                self.update_actor(game_history)

    def update_critic(self, history):
        eligibility = self.init_eligibility()
        with torch.no_grad():
            expected_values = torch.tensor(
                [
                    self.critic_model(player_sum, dealer_sum)
                    for player_sum, dealer_sum in zip(
                        history["player_sum_t"][1:], history["dealer_sum_t"][1:]
                    )
                ]
            )
            expected_values = torch.cat(
                (expected_values, torch.tensor([0], dtype=expected_values.dtype)), dim=0
            )  # No reward for terminal state
            actual_reward = torch.tensor(history["reward_t_1"])

        for player_sum, dealer_sum, actual_reward, expected_value in zip(
            history["player_sum_t"],
            history["dealer_sum_t"],
            history["reward_t_1"],
            expected_values,
        ):
            prediction = self.critic_model(player_sum, dealer_sum)
            sarsa_error = actual_reward + expected_value - prediction
            prediction.backward()
            with torch.no_grad():
                for name, param in self.critic_model.named_parameters():
                    # Discount and accumulate gradients in eligibility trace
                    eligibility[name] *= self.lambda_
                    eligibility[name] += param.grad

                    # Update weights with eligibility trace
                    param.data += (
                        self.critic_learning_rate * eligibility[name] * sarsa_error
                    )
                    # Reset gradients
                    param.grad.zero_()

    def update_actor(self, history):
        with torch.no_grad():
            predicted_rewards = torch.tensor(
                [
                    self.critic_model(player_sum, dealer_sum)
                    for player_sum, dealer_sum in zip(
                        history["player_sum_t"], history["dealer_sum_t"]
                    )
                ]
            )
            expected_value = torch.tensor(
                [
                    self.critic_model(player_sum, dealer_sum)
                    for player_sum, dealer_sum in zip(
                        history["player_sum_t"][1:], history["dealer_sum_t"][1:]
                    )
                ]
            )
            expected_value = torch.cat(
                (expected_value, torch.tensor([0], dtype=expected_value.dtype)), dim=0
            )  # No reward for terminal state
            actual_reward = torch.tensor(history["reward_t_1"])
            sarsa_errors = actual_reward + expected_value - predicted_rewards

        new_action_prob = torch.stack(
            [
                self.actor_model(player_sum, dealer_sum)[action]
                for player_sum, dealer_sum, action in zip(
                    history["player_sum_t"],
                    history["dealer_sum_t"],
                    history["action_t"],
                )
            ]
        ).squeeze(-1)
        old_action_prob = torch.tensor(history["action_probability_t"])
        ratio = new_action_prob / old_action_prob
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        surrogate_obj = ratio * sarsa_errors
        clipped_surrogate_obj = clipped_ratio * sarsa_errors
        loss = -torch.mean(torch.min(surrogate_obj, clipped_surrogate_obj))
        loss.backward()
        with torch.no_grad():
            for name, param in self.actor_model.named_parameters():
                param.data -= self.actor_learning_rate * param.grad
                param.grad.zero_()

    def init_eligibility(self):
        return {
            name: torch.zeros_like(param)
            for name, param in self.critic_model.named_parameters()
        }
