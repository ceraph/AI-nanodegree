import pickle
import random
from math import log, sqrt
from time import time
from typing import *

from isolation import Isolation
from isolation.isolation import Action
from sample_players import DataPlayer

TREE_PICKLE = 'data.pickle'
BUFFER_TIME = .025
TIME_LIMIT_IN_SEC = .150 - BUFFER_TIME
OWN_TURNS_TO_LOOK_AHEAD = 4
C_VALUE = 0.7 # 0.7 in MANGO


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
        return "{}/{}[{}]".format(self.wins, self.plays, self.state.player())

    def create_child(self, state: Isolation, causing_action) -> 'StateNode':
        child = StateNode(state, self, causing_action)
        self.children.append(child)
        return child

    @classmethod
    def create_state_tree(cls, root_node, turns=99999):
        tree = {}
        starting_depth = root_node.state.ply_count
        max_depth = starting_depth + turns * 2
        stack = [root_node]
        while stack:
            node = stack.pop()
            tree[node.state] = node
            if node.state.ply_count <= max_depth:
                stack.extend(node.children)
        print("Nodes in tree: {}".format(len(tree.keys())))
        return tree


class MonteCarloSearcher():
    __slots__ = ('_tree', '_root_node')

    def __init__(self, tree, root_node):
        self._tree = tree
        self._root_node = root_node

    def get_tree(self):
        return self._tree

    def iterate_once(self):
        leaf_node = self._selection(self._root_node)
        leaf_or_child = self._expansion(leaf_node)
        utility = self._simulation(leaf_or_child.state, leaf_or_child.state.player())
        self._backpropagation(utility, leaf_or_child)

    def _selection(self, node: StateNode) -> StateNode:
        while True:
            children = node.children
            if children:
                assert len(children) == len(node.state.actions())
                for child in children:
                    if child.plays == 0: return child

                if node.plays < 10:  # Original Paper had 30 in its game.
                    node = random.choice(children)
                else:
                    node = self._ucb1_algo(children)
            else:
                return node

    def _ucb1_algo(self, children):
        log_parent_plays = log(children[0].parent.plays)
        values = []
        for child in children:
            v = child.wins/child.plays + C_VALUE * sqrt(log_parent_plays / child.plays)
            values.append((v, child))
        best_value = max(values, key=lambda e: e[0])
        return best_value[1]

    def _expansion(self, leaf_node: StateNode) -> StateNode:
        if leaf_node.state.terminal_test(): return leaf_node
        children = self._create_children(leaf_node)
        return random.choice(children)

    def _create_children(self, parent_node: StateNode):
        for action in parent_node.state.actions():
            child_state = parent_node.state.result(action)
            child_node = parent_node.create_child(child_state, action)
            self._tree[child_state] = child_node
        return parent_node.children

    def _simulation(self, state: Isolation, leaf_player_id) -> float:
        while True:
            if state.terminal_test(): return state.utility(leaf_player_id)
            state = state.result(random.choice(state.actions()))

    def _backpropagation(self, utility: float, node: StateNode):
        leaf_player = node.state.player()  # type: int
        while node:
            node.plays += 1
            if utility == 0:
                node.wins += .5
            else:
                p = node.state.player()
                # The leaf node should lose from its parent's perspective.
                if utility < 0 and p == leaf_player or \
                        utility > 0 and p != leaf_player:
                    node.wins += 1

            if node == self._root_node:
                return
            else:
                node = node.parent


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding *named parameters
    with default values*, but the function MUST remain compatible with the
    default interface.logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """

    def get_action(self, state):
        start_time = time()
        is_first_move = state.ply_count < 2

        tree = {}
        root_node_for_turn = self._get_state_node(state, tree)

        if not is_first_move:
            tree = self.data
            assert len(tree.keys()) > 0

        runs = 0
        mcts = MonteCarloSearcher(tree, root_node_for_turn)
        while time() - start_time < TIME_LIMIT_IN_SEC:
            runs += 1
            mcts.iterate_once()
        print("MCTS ran {} times. Current player node: {}".format(runs, root_node_for_turn))

        tree = mcts.get_tree()
        self._select_action(start_time, state, root_node_for_turn)
        self._savetree(root_node_for_turn)
        print("Saved tree in {:.3}s".format(time() - start_time))
        print()

    def _select_action(self, start_time, state, root_node_for_turn):
        node = self._most_played_node(root_node_for_turn)
        self.queue.put(node.causing_action)

        print("Finished turn {} at {:.3}s. Winning node (less wins is better): {}" \
              .format(state.ply_count, time() - start_time, node))
        print("Nodes to choose from: ", end='')
        for child in node.parent.children:
            print(child, end=', ')
        print()

    def _savetree(self, root_node_for_turn):
        with open(TREE_PICKLE, 'wb') as f:
            tree = StateNode.create_state_tree(root_node_for_turn)
            pickle.dump(tree, f)

    def _most_played_node(self, root_node_for_turn) -> Action:
        children = root_node_for_turn.children
        return max(children, key=lambda e: e.plays)

    def _get_state_node(self, state, tree):
        if state in tree.keys():
            state_node = tree[state]
        else:
            state_node = self._create_root(state, tree)
        return state_node

    def _create_root(self, state: Isolation, tree):
        # assert state.ply_count <= 3, "Ply: " + str(state.ply_count)
        state_node = StateNode(state, None, None)
        tree[state] = state_node
        return state_node

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
