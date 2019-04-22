import copy
import random
from enum import Enum, unique
from math import log, sqrt
from time import time

from isolation import Isolation
from isolation.isolation import Action
from sample_players import DataPlayer
from typing import *


class StateNode:
    __slots__ = ('state', 'parent', 'children', 'wins', 'plays', 'causing_action')

    def __init__(self, state: Isolation, parent: 'StateNode', causing_action: Action):
        self.state = state  # type: Isolation
        self.parent = parent  # type: StateNode
        self.children = []  # type: List[StateNode]
        self.causing_action = causing_action  # type: Action
        self.wins = 0.0  # type: float
        self.plays = 0  # type: int

    def __str__(self):
        return "\t"*(self.state.ply_count - 1) + "{}/{}[{}]".format(self.wins, self.plays, self.state.player())

    def create_child(self, state: Isolation, causing_action) -> 'StateNode':
        child = StateNode(state, self, causing_action)
        self.children.append(child)
        return child

    @classmethod
    def create_state_tree(cls, root_node):
        tree = {}
        stack = [root_node]
        while stack:
            node = stack.pop()
            tree[node.state] = node
            stack.extend(node.children)
        return tree

@unique
class Key(Enum):
    PARENT = 0
    PLAYS = 1
    CHILDREN = 2
    CAUSING_ACTION = 3
    WINS = 4

    @staticmethod
    def create_empty_attributes():
        return {
            Key.PARENT: None,
            Key.CHILDREN: [],
            Key.CAUSING_ACTION: None,
            Key.WINS: 0.0,
            Key.PLAYS: 0,
        }


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
    *****************************add_child*****************************************
    """
    def __init__(self, player_id):
        super().__init__(player_id)
        self._tree = {} # type: Dict[Isolation, Dict]
        self._root_state_for_turn = None

    def get_action(self, state: Isolation):
        # self.queue.put(random.choice(state.actions()))  # Do something at least.
        # self.minimax_iterative_deepening(state)

        self._root_state_for_turn = state

        if state.ply_count > 1:
            assert self.context
            self._tree = self.context
            assert len(self._tree.keys()) > 0
        else:
            self._tree[state] = Key.create_empty_attributes()

        # self._root_node_for_turn = self._get_state_node(state)

        self._print_data_tree()
        # self._tree = StateNode.create_state_tree(self._root_node_for_turn)
        # self._print_data_tree()

        start = time()
        while True:
            self._monte_carlo_tree_search(state)
            if time() - start < .050:
                continue
            action = self._choose_action(state)
            print(len(self._tree.keys()))
            print("SAVING: " + str(action))
            self.queue.put(action)
            self.context = self._tree
            print("SAVED")


    def _choose_action(self, state: Isolation) -> Action:
        children = self._tree[state][Key.CHILDREN]
        most_played_state = max(children, key=lambda e: self._tree[e][Key.PLAYS])
        return self._tree[most_played_state][Key.CAUSING_ACTION]

    def _monte_carlo_tree_search(self, state: Isolation):
        leaf_state = self._mcts_selection(state)
        leaf_or_child_state = self._mcts_expansion(leaf_state)
        utility = self._mcts_simulation(leaf_or_child_state, leaf_or_child_state.player())
        self._mcts_backprop(utility, leaf_or_child_state)

    def _mcts_selection(self, state: Isolation) -> Isolation:
        while True:
            if self._tree[state][Key.CHILDREN]:
                children = self._tree[state][Key.CHILDREN] # type: List[Isolation]
                assert len(children) == len(state.actions()), \
                    "Children: {}, Actions: {}".format(len(children), len(state.actions()))
                for child in children:
                    if self._tree[child][Key.PLAYS] == 0: return child

                if self._tree[state][Key.PLAYS] < 15: # Original Paper had 30 in its game.
                    state = random.choice(children)
                else:
                    state = self._ucb1_algo(state, children)
            else:
                return state

    def _ucb1_algo(self, parent_state: Isolation, children: List[Isolation]) -> Isolation:
        log_parent_plays = log(self._tree[parent_state][Key.PLAYS])
        # is_own_move = parent_state.player() == self.player_id
        c = sqrt(2)
        values = []
        for child in children:
            relative_wins = self._tree[child][Key.WINS] / self._tree[child][Key.PLAYS]
            v = relative_wins + c * sqrt(log_parent_plays / self._tree[child][Key.PLAYS])
            values.append((v, child))
        best_value = max(values, key=lambda e: e[0])
        return best_value[1]

    def _mcts_expansion(self, leaf_state: Isolation) -> Isolation:
        if leaf_state.terminal_test(): return leaf_state
        children = self._create_children(leaf_state)
        return random.choice(children)

    def _create_children(self, parent_state: Isolation):
        child_states = []
        for action in parent_state.actions():
            child_state = parent_state.result(action)
            child_states.append(child_state)

            child_vals = Key.create_empty_attributes()
            child_vals[Key.PARENT] = parent_state
            child_vals[Key.CAUSING_ACTION] = action
            self._tree[child_state] = child_vals
        self._tree[parent_state][Key.CHILDREN] = child_states
        return child_states

    def _mcts_simulation(self, state: Isolation, leaf_player_id) -> float:
        while True:
            if state.terminal_test(): return state.utility(leaf_player_id)
            state = state.result(random.choice(state.actions()))

    def _mcts_backprop(self, utility: float, leaf_state: Isolation):
        leaf_player = leaf_state.player()  # type: int
        state = leaf_state
        while state:
            state_vals = self._tree[state]
            state_vals[Key.PLAYS] += 1
            if utility == 0:
                state_vals[Key.WINS] += 1
            else:
                p = state.player()
                if utility < 0 and p != leaf_player or \
                   utility > 0 and p == leaf_player:
                    state_vals[Key.WINS] += 1

            if state == self._root_state_for_turn: # TODO: Remove if tree gets always pruned.
                return
            else:
                state = state_vals[Key.PARENT]

    def _print_data_tree(self):
        stack = [self._root_state_for_turn]
        while stack:
            state = stack.pop()
            print(state)

            children = self._tree[state][Key.CHILDREN]
            if children:
                stack.extend(children)

    def minimax_iterative_deepening(self, state):
        alpha = float("-inf")
        beta = float("inf")
        depth = 1
        while True:
            best_action_so_far = self._minimax_with_alpha_beta_pruning(state, depth, alpha, beta)
            self.queue.put(best_action_so_far)
            depth += 1

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
