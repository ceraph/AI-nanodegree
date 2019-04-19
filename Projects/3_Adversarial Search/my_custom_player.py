from isolation import Isolation
from isolation.isolation import Action
from sample_players import DataPlayer
from typing import *


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
        import random
        self.queue.put(random.choice(state.actions()))

        alpha = float("-inf")
        beta = float("inf")
        depth = 3
        minimax = self._minimax_with_alpha_beta_pruning(state, depth, alpha, beta)
        self.queue.put(minimax)

    def _minimax_with_alpha_beta_pruning(self, state, depth, alpha, beta) -> Action:

        def min_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self._heuristic_nr_of_moves(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), depth - 1, alpha, beta))
                if value <= alpha:
                    break
                beta = min(beta, value)
            return value

        def max_value(state, depth, alpha, beta):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self._heuristic_nr_of_moves(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), depth - 1, alpha, beta))
                if value >= beta:
                    break
                alpha = max(alpha, value)
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), depth - 1, alpha, beta))

    def _heuristic_nr_of_moves(self, state):
        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)
        return len(own_liberties) - len(opp_liberties)
