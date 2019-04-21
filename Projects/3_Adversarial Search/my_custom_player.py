import copy
import random
from math import log, sqrt

from isolation import Isolation
from isolation.isolation import Action
from sample_players import DataPlayer
from typing import *

ASSUMED_NEXT_STATES = 'assumed_next_states'

ROOT = 'root'

TREE = 'tree'

CORRUPTED_PARENT = 'corrupted_parent'


class StateNode:
    def __init__(self, state: Isolation, parent: Union['StateNode',None], causing_action, player_id):
        self._state = state
        self._parent = parent
        self._causing_action = causing_action
        self._children = []
        self._level = 0 if parent is None else parent.get_level() + 1
        self._player_id = player_id
        self.wins = 0.0
        self.plays = 0

    def create_child(self, state: Isolation, causing_action, player_id) -> 'StateNode':
        child = StateNode(state, self, causing_action, player_id)
        self._children.append(child)
        return child

    def get_children(self) -> Tuple['StateNode', ...]:
        return tuple(self._children)  # Make immutable.

    def clear_children(self):
        self._children = []

    def get_causing_action(self):
        return self._causing_action

    def get_state(self):
        return self._state

    def get_parent(self) -> 'StateNode':
        return self._parent

    def get_player_id(self) -> int:
        return self._player_id

    def get_level(self):
        return self._level

    def __str__(self):
        return "\t"*self._level + "{}/{}[{}]".format(self.wins, self.plays, self._player_id)


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
    class Data:
        def __init__(self):
            self.tree = {}
            self.root = None
            self.assumed_next_states = []
    
    def __init__(self, player_id):
        super().__init__(player_id)
        self._data = self.Data()
        self._root_node_for_turn = None

    def get_action(self, state: Isolation):
        # self.queue.put(random.choice(state.actions()))  # Do something at least.
        # self.minimax_iterative_deepening(state)

        if self.context:
            self._data = copy.deepcopy(self.context)

            states = self._data.assumed_next_states
            assert state.board in [state.board for state in states], states

            assert len(self._data.tree.keys()) > 0

        self._root_node_for_turn = self._get_state_node(state)
        action = self._choose_action()
        self.queue.put(action) # Do something at least.

        self._print_data_tree()
        assert self._root_node_for_turn.get_player_id() == self.player_id
        # print(len(self.tree.keys()))
        while True:
            self._monte_carlo_tree_search(self._root_node_for_turn)
            action = self._choose_action()
            self.queue.put(action)
            self.context = copy.deepcopy(self._data)
            
    def _choose_action(self):
        children = self._root_node_for_turn.get_children()
        if not children:
            self._data.assumed_next_states = self._root_node_for_turn.get_state().actions()
            return random.choice(self._root_node_for_turn.get_state().actions())
        most_played_node = max(children, key=lambda e: e.plays)
        action = most_played_node.get_causing_action()
        self._data.assumed_next_states = [child.get_state() for child in most_played_node.get_children()]
        return action

    def _get_state_node(self, state):
        if state in self._data.tree.keys():
            state_node = self._data.tree[state]
        else:
            state_node = self._create_root(state)
        return state_node

    def _create_root(self, state: Isolation):
        assert state.ply_count <= 1, "Ply: " + str(state.ply_count)
        assert self._data.root is None
        state_node = StateNode(state, None, None, self.player_id)
        self._data.root = state_node
        self._data.tree[state] = state_node
        return state_node

    def _monte_carlo_tree_search(self, node: StateNode):
        leaf_node = self._mcts_selection(node)
        leaf_or_child = self._mcts_expansion(leaf_node)
        utility = self._mcts_simulation(leaf_or_child.get_state(), leaf_or_child.get_player_id())
        self._mcts_backprop(utility, leaf_or_child)

    def _mcts_selection(self, node: StateNode) -> StateNode:
        while True:
            children = node.get_children()
            if children:
                assert len(children) == len(node.get_state().actions())
                for child in children:
                    if child.plays == 0: return child
                assert not [child for child in children if child.plays < 1]
                node = self._ucb1_algo(children)
            else:
                return node

    def _ucb1_algo(self, children):
        c = sqrt(2)
        log_parent_plays = log(children[0].get_parent().plays)
        is_own_move = children[0].get_player_id() == self.player_id
        values = []
        for child in children:
            v = child.wins/child.plays + c * sqrt(log_parent_plays / child.plays)
            values.append((v, child))
        if is_own_move:
            best_value = max(values, key=lambda e: e[0])
        else:
            best_value = min(values, key=lambda e: e[0])
        return best_value[1]

    def _mcts_expansion(self, leaf_node: StateNode) -> StateNode:
        if leaf_node.get_state().terminal_test(): return leaf_node
        children = self._create_children(leaf_node)
        return random.choice(children)

    def _create_children(self, parent_node: StateNode):
        for action in parent_node.get_state().actions():
            child_state = parent_node.get_state().result(action)
            child_node = parent_node.create_child(child_state, action, parent_node.get_player_id() ^ 1)
            self._data.tree[child_state] = child_node
        return parent_node.get_children()

    def _mcts_simulation(self, state: Isolation, leaf_player_id) -> float:
        while True:
            if state.terminal_test(): return state.utility(leaf_player_id)
            state = state.result(random.choice(state.actions()))

    def _mcts_backprop(self, utility: float, node: StateNode):
        leaf_player_id = node.get_player_id()  # type: int
        while node:
            assert(node.plays, sum([child.plays for child in node.get_children()]))
            node.plays += 1
            if utility == 0:
                node.wins += .5
            else:
                if utility < 0 and node.get_player_id() != leaf_player_id or \
                   utility > 0 and node.get_player_id() == leaf_player_id:
                    node.wins += 1

            if node == self._root_node_for_turn:
                return
            else:
                node = node.get_parent()

    def _print_data_tree(self):
        root = self._data.root  # type: StateNode
        if not root: return

        stack = [root]
        while stack:
            node = stack.pop()
            print(node)

            children = node.get_children()
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
