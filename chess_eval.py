from abc import ABC, abstractmethod
import chess
import chess.pgn
import numpy as np
import random

class MCTSState(ABC):
    @abstractmethod
    def get_actions(self):
        '''Get the list of possible actions to take in this state.

        Returns:
            list: List of actions.
        '''
        pass

    @abstractmethod
    def apply_action(self, action):
        '''Choose an action from this state.

        Returns:
            MCTSState: The next state after applying the action.
        '''
        pass

    @abstractmethod
    def is_terminal(self):
        '''Check if this is a terminal state.

        Returns:
            bool
        '''
        pass

    @abstractmethod
    def reward(self):
        '''Get the reward for this state.

        Returns:
            number or arraylike
        '''
        pass

    def rollout(self, max_steps):
        '''Perform a rollout from this state.

        Args:
            max_steps (int): Maximum number of steps to take.
        
        Returns:
            number or arraylike: Reward at the end of the rollout.
        '''
        state = self

        num_steps = 0

        while num_steps < max_steps:
            if state.is_terminal():
                break

            state = state.apply_action(random.choice(state.get_actions()))
            num_steps += 1

        reward = state.reward()
        return reward

    @abstractmethod
    def process_value(self, value):
        '''When choosing which child node to visit during tree traversal, this
        method is used to process the value of each child node. For
        instance, if the value has N elements, one for each player, and
        the game is turn-based, this method should pick out the element for the
        player whose turn it is.

        With only one reward element, ``value`` can simply be returned.

        Args:
            value (number or arraylike): Value for the tree node, whose
                shape is the same as that returned by :meth:`~.reward`.

        Returns:
            number
        '''
        pass

class ChessState(MCTSState):
    def __init__(self, board):
        self.board = board

    def get_actions(self):
        return list(self.board.legal_moves)

    def apply_action(self, action):
        board = self.board.copy()
        board.push(action)
        return ChessState(board)

    def is_terminal(self):
        return self.board.is_game_over() | self.board.is_fifty_moves()

    def reward(self):
        if self.board.is_checkmate():
            winner = not self.board.turn
            if winner == chess.WHITE:
                return np.array([1., 0.])
            else:
                return np.array([0., 1.])
        else:
            return np.array([0., 0.])

    def process_value(self, value):
        if self.board.turn == chess.WHITE:
            return value[0]
        else:
            return value[1]

class MCTSNode:
    '''Node of an MCTS tree.

    Args:
        state (MCTSState): The state associated with this node.

        value_shape (tuple): Shape of a node's value.

        parent (MCTSNode or None, optional): The parent node of this node.
            Default: ``None``
    '''
    def __init__(self, state, value_shape, parent=None):
        self.state = state
        self.parent = parent
        self.num_visits = 0
        self.action_child_map = {}
        self.value_sum = np.zeros(value_shape)

    def value_average(self):
        if self.num_visits == 0:
            return 0.0
        else:
            return self.parent.state.process_value(self.value_sum) / self.num_visits

    C = np.sqrt(2.0)

    def traversal_priority(self):
        if self.num_visits == 0:
            ucb1 = float('inf')
        else:
            assert self.parent.num_visits > 0
            ucb1 = self.value_average() + self.C * np.sqrt(np.log(self.parent.num_visits) / self.num_visits)
        return ucb1

    def get_action_value_pairs(self):
        return [(action, child.value_average()) for action, child in self.action_child_map.items()]

    def is_leaf(self):
        return len(self.action_child_map) == 0

    def __str__(self):
        string = (
            f'id: {id(self)}\n'
            f'  num_visits: {self.num_visits}\n'
            f'  value_sum: {self.value_sum}\n'
            f'  value_average: {self.value_average()}\n'
            f'  actions: {[self.state.board.san(action) for action in self.action_child_map.keys()]}\n'
            f'  children: {[id(child) for child in self.action_child_map.values()]}\n'
            f'  children_value_averages: {[child.value_average() for child in self.action_child_map.values()]}\n'
            f'  turn: {"white" if self.state.board.turn == chess.WHITE else "black"}'
        )
        for child in self.action_child_map.values():
            string += '\n' + str(child)
        return string

def traverse_MCTS(tree, max_rollout_steps):
    '''Performs one update step of an MCTS tree.

    Uses traversal algorithm described in:

        * https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
        * https://www.youtube.com/watch?v=UXW2yZndl7U

    Args:
        tree (MCTSNode): Root node of MCTS tree.

        max_rollout_steps (int): Maximum number of steps to take for each rollout.
    '''
    current_node = tree
    done = False

    while not done:
        if current_node.is_leaf():
            if (current_node.num_visits > 0 and not current_node.state.is_terminal()) or current_node.parent is None:
                actions = current_node.state.get_actions()

                for action in actions:
                    current_node.action_child_map[action] = MCTSNode(current_node.state.apply_action(action), current_node.value_sum.shape, current_node)

                chosen_idx = random.choice(range(len(actions)))
                current_node = list(current_node.action_child_map.values())[chosen_idx]

            rollout_reward = current_node.state.rollout(max_steps=max_rollout_steps)
            done = True

        else:
            actions = list(current_node.action_child_map.keys())
            children = list(current_node.action_child_map.values())
            priorities = [child.traversal_priority() for child in children]
            chosen_idx = np.argmax(priorities)
            current_node = children[chosen_idx]

    while current_node is not None:
        current_node.num_visits += 1
        current_node.value_sum += rollout_reward
        current_node = current_node.parent


def eval_fen(fen, iters, max_rollout_steps):
    '''Evaluate a chess position.

    Args:
        fen (str): FEN string rerpresenting the a chess board position.

        iters (int): Number of iterations to run Monte Carlo tree search.

        max_rollout_steps (int): Maximum number of steps to take for each rollout.

    Returns:
        list of tuples: Each tuple contains one of the legal moves in san
            format, paired with its calculated value.
    '''
    board = chess.Board(fen)
    tree = MCTSNode(ChessState(board), (2,))

    for _ in range(iters):
        traverse_MCTS(tree, max_rollout_steps)

    action_values = reversed(sorted(tree.get_action_value_pairs(), key=lambda x: x[1]))

    return [(board.san(action), value) for action, value in action_values]

if __name__ == '__main__':
    # White has M1, best move Rd8. Any other moves lose to M2 or M1.
    print(eval_fen('7k/6pp/7p/7K/8/8/6q1/3R4 w - - 0 1', 100, 20)[0])

    # Black has M1, best move Qg6#. Other moves give rough equality or worse.
    print(eval_fen('6qk/2R4p/7K/8/8/8/8/4R3 b - - 1 1', 100, 20)[0])

    # White has M2, best move Rxg8+. Any other move loses.
    print(eval_fen('2R3qk/5p1p/7K/8/8/8/5r2/2R5 w - - 0 1', 10_000, 20)[0])

    # White has M3, best move is N7h6+. Any other move loses or ties.
    print(eval_fen('qrr3k1/5Npp/8/3Q1N2/8/8/5PPP/6K1 w - - 0 1', 20_000, 20)[0])

    # White has M4, best move is Nf7+. Any other move loses
    # This one takes ~7 minutes on my machine.
    #print(eval_fen('qrr4k/6pp/8/3QNN2/8/8/5PPP/6K1 w - - 0 1', 400_000, 20))

