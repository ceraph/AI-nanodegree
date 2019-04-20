import random

from isolation import Isolation
from isolation.isolation import Action
from sample_players import DataPlayer
from typing import *

class StateNode:
    def __init__(self, state: Isolation, previous_node):
        self._state = state
        self._parent = previous_node
        self._children = []
        self.player = state.player()
        self.wins = 0
        self.plays = 0

    def get_state(self):
        return self._state

    def get_parent(self):
        return self._parent

    def get_children(self):
        return self._children

class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding *named parameters
    with default values*, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state: Isolation):
        # Do something at least.
        self.queue.put(random.choice(state.actions()))

        # Iterative Deepening
        # alpha = float("-inf")
        # beta = float("inf")
        # depth = 1
        # while True:
        #     best_action_so_far = self._minimax_with_alpha_beta_pruning(state, depth, alpha, beta)
        #     self.queue.put(best_action_so_far)
        #     depth += 1

        # Monte Carlo Tree Search
        state_node = self.context[state]
        if not state_node:
            state_node = StateNode(state, None) # Create root node.

        leaf_node = self._mcts_selection(state_node)
        child_node = self._mcts_expansion(leaf_node)
        utility = self._mcts_simulation(child_node.get_state())
        self._mcts_backprop(utility, child_node)

    def _mcts_selection(self, state_node: StateNode) -> StateNode:
        while True:
            children = state_node.get_children()
            if children:
                state_node = random.choice(state_node.get_children())
            else:
                return state_node

    def _mcts_expansion(self, parent_node: StateNode) -> StateNode:
        a = random.choice(parent_node.get_state().actions())
        new_state = parent_node.get_state().result(a)
        return StateNode(new_state, parent_node)

    def _mcts_simulation(self, state: Isolation):
        while True:
            if state.terminal_test(): return state.utility(self.player_id)
            state = state.result(random.choice(state.actions()))

    def _mcts_backprop(self, utility, node: StateNode):
        while node:
            node.plays += 1

            if utility == 0:
                node.wins += .5
            elif utility > 0 and node.player == self.player_id():
                node.wins += 1

            node = node.get_parent()

    def _minimax_with_alpha_beta_pruning(self, state, depth, alpha, beta) -> Action:

        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self._evaluate(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                if value <= alpha:
                    break
                beta = min(beta, value)
            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self._evaluate(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                if value >= beta:
                    break
                alpha = max(alpha, value)
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1, alpha, beta))

    def _evaluate(self, state):
        return self._heuristic_nr_of_moves(state)

    def _heuristic_nr_of_moves(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)

