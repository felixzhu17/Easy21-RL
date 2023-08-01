import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from game import Action
import torch
import torch.nn as nn
import torch.nn.functional as F


class Easy21StateValueApproximation(nn.Module):
    VALID_PLAYER_SUMS = list(range(22))
    VALID_DEALER_SUMS = list(range(11))
    ACTIONS = list(Action)
    PLAYER_BINS = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 22)]
    DEALER_BINS = [(1, 4), (4, 7), (7, 11)]

    def __init__(self):
        super().__init__()
        self.feature_size = len(self.PLAYER_BINS) * len(self.DEALER_BINS)
        self.fc1 = nn.Linear(self.feature_size, 1, bias=False)

    def forward(self, player_sum, dealer_sum):
        x = self.get_feature(player_sum, dealer_sum)
        x = self.fc1(x)
        return torch.tanh(x)

    def get_feature(self, player_sum, dealer_sum):
        for i, d_bin in enumerate(self.DEALER_BINS):
            if d_bin[0] <= dealer_sum < d_bin[1]:
                dealer_feature = i

        for i, p_bin in enumerate(self.PLAYER_BINS):
            if p_bin[0] <= player_sum < p_bin[1]:
                player_feature = i

        feature_vector = torch.zeros(len(self.PLAYER_BINS), len(self.DEALER_BINS))
        feature_vector[player_feature, dealer_feature] = 1.0
        return feature_vector.flatten()

    def get_value_function(self):
        output_array = np.zeros(
            (
                len(self.VALID_PLAYER_SUMS) - 1,
                len(self.VALID_DEALER_SUMS) - 1,
            )
        )
        # Perform the forward pass for all combinations
        with torch.no_grad():  # Ensure no gradient computation is made
            for player in range(1, len(self.VALID_PLAYER_SUMS)):
                for dealer in range(1, len(self.VALID_DEALER_SUMS)):
                    output = self.forward(player, dealer)
                    output_array[
                        player - 1, dealer - 1
                    ] = output.detach().numpy()  # Convert tensor to value and store
        return output_array

    def plot_value_function(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        state_value = self.get_value_function()
        # Generate coordinate matrices, starting from index 1
        X, Y = np.meshgrid(
            range(1, state_value.shape[1] + 1), range(1, state_value.shape[0] + 1)
        )

        # Plot the surface, starting from index 1
        ax.plot_surface(X, Y, state_value, cmap="viridis")

        # Setting labels
        ax.set_xlabel("Dealer Sum")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")
        ax.set_title("Value Function")

        # Displaying the plot
        plt.show()


class Easy21PolicyApproximation(nn.Module):
    VALID_PLAYER_SUMS = list(range(22))
    VALID_DEALER_SUMS = list(range(11))
    ACTIONS = list(Action)
    PLAYER_BINS = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 22)]
    DEALER_BINS = [(1, 4), (4, 7), (7, 11)]

    def __init__(self):
        super().__init__()
        self.feature_size = len(self.PLAYER_BINS) * len(self.DEALER_BINS)
        self.action_space = len(self.ACTIONS)
        self.fc1 = nn.Linear(self.feature_size, self.action_space, bias=False)

    def forward(self, player_sum, dealer_sum):
        x = self.get_feature(player_sum, dealer_sum)
        x = self.fc1(x)
        return F.softmax(x, dim=0)

    def get_feature(self, player_sum, dealer_sum):
        for i, d_bin in enumerate(self.DEALER_BINS):
            if d_bin[0] <= dealer_sum < d_bin[1]:
                dealer_feature = i

        for i, p_bin in enumerate(self.PLAYER_BINS):
            if p_bin[0] <= player_sum < p_bin[1]:
                player_feature = i

        feature_vector = torch.zeros(len(self.PLAYER_BINS), len(self.DEALER_BINS))
        feature_vector[player_feature, dealer_feature] = 1.0
        return feature_vector.flatten()

    def get_epsilon_greedy_action(self, player_sum, dealer_sum, epsilon=0.05):
        if random.random() < epsilon:
            return random.choice(self.ACTIONS)
        else:
            return self.get_optimal_action(player_sum, dealer_sum)

    def get_optimal_action(self, player_sum, dealer_sum):
        # Use no_grad to prevent tracking of gradients during this forward pass
        with torch.no_grad():
            action_probs = self.forward(player_sum, dealer_sum)
        # Find the indices of maximum elements. This will return both values and indices.
        # Since we need only indices, we select those.
        max_probs, max_indices = torch.max(action_probs, dim=0)
        # max_indices is a tensor. We need to convert it to a python list to find all occurrences of max_probs.
        max_indices = max_indices.tolist()
        # Get all actions with maximum probability
        optimal_actions = [
            Action(i) for i, prob in enumerate(action_probs) if prob == max_probs
        ]
        # If there are multiple optimal actions, pick one uniformly at random
        optimal_action = random.choice(optimal_actions)
        return optimal_action


class Easy21ActionValueApproximation(nn.Module):
    VALID_PLAYER_SUMS = list(range(22))
    VALID_DEALER_SUMS = list(range(11))
    ACTIONS = list(Action)
    PLAYER_BINS = [(1, 6), (4, 9), (7, 12), (10, 15), (13, 18), (16, 22)]
    DEALER_BINS = [(1, 4), (4, 7), (7, 11)]

    def __init__(self):
        super().__init__()
        self.feature_size = (
            len(self.PLAYER_BINS) * len(self.DEALER_BINS) * len(self.ACTIONS)
        )
        self.fc1 = nn.Linear(self.feature_size, 1, bias=False)

    def forward(self, player_sum, dealer_sum, action):
        x = self.get_feature(player_sum, dealer_sum, action)
        x = self.fc1(x)
        return torch.tanh(x)

    def get_feature(self, player_sum, dealer_sum, action):
        for i, d_bin in enumerate(self.DEALER_BINS):
            if d_bin[0] <= dealer_sum < d_bin[1]:
                dealer_feature = i

        for i, p_bin in enumerate(self.PLAYER_BINS):
            if p_bin[0] <= player_sum < p_bin[1]:
                player_feature = i

        feature_vector = torch.zeros(
            len(self.PLAYER_BINS), len(self.DEALER_BINS), len(self.ACTIONS)
        )
        feature_vector[player_feature, dealer_feature, action] = 1.0
        return feature_vector.flatten()

    def get_epsilon_greedy_action(self, player_sum, dealer_sum, epsilon=0.05):
        if random.random() < epsilon:
            return random.choice(self.ACTIONS)
        else:
            return self.get_optimal_action(player_sum, dealer_sum)

    def get_optimal_action(self, player_sum, dealer_sum):
        with torch.no_grad():
            action_values = {
                action: self.forward(player_sum, dealer_sum, action.value)
                for action in self.ACTIONS
            }
            max_value = max(action_values.values())
        # Filter the actions that have the max value
        max_actions = [
            action for action, value in action_values.items() if value == max_value
        ]
        # Choose one of the actions with max value randomly
        return random.choice(max_actions)

    def get_value_function(self):
        output_array = np.zeros(
            (
                len(self.VALID_PLAYER_SUMS) - 1,
                len(self.VALID_DEALER_SUMS) - 1,
                len(self.ACTIONS),
            )
        )
        # Perform the forward pass for all combinations
        with torch.no_grad():  # Ensure no gradient computation is made
            for player in range(1, len(self.VALID_PLAYER_SUMS)):
                for dealer in range(1, len(self.VALID_DEALER_SUMS)):
                    for action in range(len(self.ACTIONS)):
                        output = self.forward(player, dealer, action)
                        output_array[
                            player - 1, dealer - 1, action
                        ] = output.detach().numpy()  # Convert tensor to value and store
        return output_array.max(axis=2)

    def plot_value_function(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        state_value = self.get_value_function()
        # Generate coordinate matrices, starting from index 1
        X, Y = np.meshgrid(
            range(1, state_value.shape[1] + 1), range(1, state_value.shape[0] + 1)
        )

        # Plot the surface, starting from index 1
        ax.plot_surface(X, Y, state_value, cmap="viridis")

        # Setting labels
        ax.set_xlabel("Dealer Sum")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")
        ax.set_title("Value Function")

        # Displaying the plot
        plt.show()


class Easy21Policy:
    VALID_PLAYER_SUMS = list(range(22))
    VALID_DEALER_SUMS = list(range(11))
    ACTIONS = list(Action)

    def __init__(self, N0=100):
        self.N0 = N0
        self.action_state_value = np.zeros(
            (len(self.VALID_PLAYER_SUMS), len(self.VALID_DEALER_SUMS), len(Action))
        )
        self.action_state_counts = np.zeros(
            (len(self.VALID_PLAYER_SUMS), len(self.VALID_DEALER_SUMS), len(Action))
        )

    @property
    def state_value(self):
        return self.action_state_value.max(axis=2)

    def get_action_state_value(self, player_sum, dealer_sum, action):
        return self.action_state_value[player_sum, dealer_sum, action]

    def get_action_state_counts(self, player_sum, dealer_sum, action):
        return self.action_state_counts[player_sum, dealer_sum, action]

    def get_state_counts(self, player_sum, dealer_sum):
        return self.action_state_counts[player_sum, dealer_sum].sum()

    def get_state_value(self, player_sum, dealer_sum):
        return self.state_value[player_sum, dealer_sum]

    def get_epsilon(self, player_sum, dealer_sum):
        return self.N0 / (self.N0 + self.get_state_counts(player_sum, dealer_sum))

    def get_epsilon_greedy_action(self, player_sum, dealer_sum, epsilon=None):
        if epsilon is None:
            epsilon = self.get_epsilon(player_sum, dealer_sum)
        if random.random() < epsilon:
            return random.choice(self.ACTIONS)
        else:
            return self.get_optimal_action(player_sum, dealer_sum)

    def get_optimal_action(self, player_sum, dealer_sum):
        action_values = self.action_state_value[player_sum, dealer_sum]
        max_value = np.max(action_values)
        # Get the indices of all actions that have the maximum value
        optimal_actions = np.where(action_values == max_value)[0]
        # Randomly select one of the optimal actions
        chosen_action = np.random.choice(optimal_actions)
        return Action(chosen_action)

    def get_update_alpha(self, player_sum, dealer_sum, action):
        action_state_count = self.get_action_state_counts(
            player_sum, dealer_sum, action
        )
        if action_state_count == 0:  # Guard clause to prevent ZeroDivisionError
            return 0
        return 1 / (action_state_count + 1)

    def plot_value_function(self):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Generate coordinate matrices, starting from index 1
        X, Y = np.meshgrid(
            range(1, self.state_value.shape[1]), range(1, self.state_value.shape[0])
        )

        # Plot the surface, starting from index 1
        ax.plot_surface(X, Y, self.state_value[1:, 1:], cmap="viridis")

        # Setting labels
        ax.set_xlabel("Dealer Sum")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")
        ax.set_title("Value Function")

        # Displaying the plot
        plt.show()
