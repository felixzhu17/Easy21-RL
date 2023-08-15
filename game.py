import random
from enum import Enum
import copy
import pandas as pd


class Action(Enum):
    HIT = 0
    STICK = 1


class GameOverError(Exception):
    pass


class GameState:
    def __init__(self, player_sum, dealer_sum, terminal):
        self.player_sum = player_sum
        self.dealer_sum = dealer_sum
        self.terminal = terminal

    def __str__(self):
        return f"Player Sum: {self.player_sum}, Dealer Sum: {self.dealer_sum}, Terminal: {self.terminal}"


class StateHistory:
    def __init__(self, t, state_t, action_t, reward_t_1, action_probability_t=None):
        self.t = t
        self.state_t = state_t
        self.action_t = action_t
        self.reward_t_1 = reward_t_1
        self.action_probability_t = action_probability_t

    def __str__(self):
        return f"t: {self.t}, state_t: {self.state_t}, action_t: {self.action_t}, reward_t_1: {self.reward_t_1}, action_probability_t: {self.action_probability_t}"

    def to_dict(self):
        return {
            "t": self.t,
            "player_sum_t": self.state_t.player_sum,
            "dealer_sum_t": self.state_t.dealer_sum,
            "action_t": self.action_t.value,
            "reward_t_1": self.reward_t_1,
            "action_probability_t": self.action_probability_t,
        }


class Easy21:
    def __init__(self, verbose=False):
        self.state = GameState(random.randint(1, 10), random.randint(1, 10), False)
        self.history = []
        self.t = 0
        self.verbose = verbose
        if self.verbose:
            print(self.state)

    def step(self, action, action_probability=None):
        self.check_game_over()
        self.validate_action(action)

        start_state = self.get_state()
        if self.verbose:
            print(action)

        if action == Action.HIT:
            self.state.player_sum += self.draw_card()
            if self.state.player_sum > 21 or self.state.player_sum < 1:
                self.state.player_sum = 0  # Busted
                self.state.terminal = True
                state_update = StateHistory(
                    t=self.t,
                    state_t=start_state,
                    action_t=Action.HIT,
                    reward_t_1=-1,
                    action_probability_t=action_probability,
                )
            else:
                state_update = StateHistory(
                    t=self.t,
                    state_t=start_state,
                    action_t=Action.HIT,
                    reward_t_1=0,
                    action_probability_t=action_probability,
                )

        else:
            while self.state.dealer_sum < 17 and self.state.dealer_sum > 0:
                self.state.dealer_sum += self.draw_card()
            self.state.terminal = True
            if self.state.dealer_sum > 21 or self.state.dealer_sum < 1:
                self.state.dealer_sum = 0  # Busted
                state_update = StateHistory(
                    t=self.t,
                    state_t=start_state,
                    action_t=Action.STICK,
                    reward_t_1=1,
                    action_probability_t=action_probability,
                )
            if self.state.player_sum > self.state.dealer_sum:
                state_update = StateHistory(
                    t=self.t,
                    state_t=start_state,
                    action_t=Action.STICK,
                    reward_t_1=1,
                    action_probability_t=action_probability,
                )
            elif self.state.player_sum == self.state.dealer_sum:
                state_update = StateHistory(
                    t=self.t,
                    state_t=start_state,
                    action_t=Action.STICK,
                    reward_t_1=0,
                    action_probability_t=action_probability,
                )
            else:
                state_update = StateHistory(
                    t=self.t,
                    state_t=start_state,
                    action_t=Action.STICK,
                    reward_t_1=-1,
                    action_probability_t=action_probability,
                )

        self.history.append(state_update)
        self.t += 1
        if self.verbose:
            print(self.state)

        return state_update

    def draw_card(self):
        card = random.randint(1, 10)
        if random.random() < 1 / 3:  # Red cards are 1/3 of the deck
            card = -card
        if self.verbose:
            print(f"Drew {card}")
        return card

    def get_state(self):
        return copy.deepcopy(self.state)

    def get_history(self):
        df = pd.DataFrame([i.to_dict() for i in self.history])
        df = df.sort_values(by="t", ascending=False)

        # Compute cumulative sum of 'reward_t_1'
        df["cumulative_reward"] = df["reward_t_1"].cumsum()
        return df

    def validate_action(self, action):
        if not isinstance(action, Action):
            raise ValueError("Invalid action")

    def check_game_over(self):
        if self.state.terminal:
            raise GameOverError("Game is over, please reset")
