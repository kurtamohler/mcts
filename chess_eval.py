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

        for _ in range(max_steps):
            if state.is_terminal():
                break

            state = state.apply_action(random.choice(state.get_actions()))

        reward = state.reward()
        return reward

    @abstractmethod
    def traversal_value(self, average_value):
        '''When choosing which child node to visit during tree traversal, this
        method is used to process the average value of each child node. For
        instance, if the average value has N elements, one for each player, and
        the game is turn-based, this method should pick out the element for the
        player whose turn it is.

        For a one-player game, most likely ``average_value`` can simply be returned.

        Args:
            average_value (number or arraylike): Average value for the tree node.

        Returns:
            number or arraylike
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
            return 1 if winner == chess.WHITE else -1
        else:
            return 0

    def traversal_value(self, average_value):
        # If it's black's turn, a negative value average is good, and positive is bad, so we
        # need to flip it.
        value_sign = 1 if self.board.turn == chess.WHITE else -1
        return value_sign * average_value

class MCTSNode:
    '''Node of an MCTS tree.

    Args:
        state (MCTSState): The state associated with this node.

        parent (MCTSNode or None): The parent node of this node.
    '''
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.num_visits = 0
        self.action_child_map = {}
        self.value_sum = 0.0

    def value_average(self):
        if self.num_visits == 0:
            return 0.0
        else:
            return self.value_sum / self.num_visits

    def traversal_priority(self):
        C = 200.0
        ucb1 = self.state.traversal_value(self.value_average()) + C * np.sqrt(np.log(self.parent.num_visits) / self.num_visits)
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

    Uses traversal algorithm described in: https://www.youtube.com/watch?v=UXW2yZndl7U

    Args:
        tree (MCTSNode): Root node of MCTS tree.

        max_rollout_steps (int): Maximum number of steps to take for each rollout.
    '''
    current_node = tree

    done = False

    spacer = ''

    while not done:
        if current_node.is_leaf():
            if (current_node.num_visits > 0 and not current_node.state.is_terminal()) or current_node.parent is None:
                actions = current_node.state.get_actions()

                #print(f"{spacer}adding child actions: {[current_node.state.board.san(action) for action in actions]}")

                for action in actions:
                    current_node.action_child_map[action] = MCTSNode(current_node.state.apply_action(action), current_node)

                chosen_idx = random.choice(range(len(actions)))

                #print(f"{spacer}choosing {current_node.state.board.san(actions[chosen_idx])}")

                current_node = list(current_node.action_child_map.values())[chosen_idx]

            rollout_reward = current_node.state.rollout(max_steps=max_rollout_steps)

            #print(f"{spacer}rollout reward {rollout_reward}")

            done = True

        else:
            actions = list(current_node.action_child_map.keys())
            children = list(current_node.action_child_map.values())
            priorities = [child.traversal_priority() for child in children]
            chosen_idx = np.argmax(priorities)

            #print(f"{spacer}choosing {current_node.state.board.san(actions[chosen_idx])}")

            current_node = children[chosen_idx]
            spacer = '-' + spacer

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
    tree = MCTSNode(ChessState(board))

    for _ in range(iters):
        traverse_MCTS(tree, max_rollout_steps)

    #print('------------------------------')
    #print(tree)
    #print('------------------------------')

    action_values = reversed(sorted(tree.get_action_value_pairs(), key=lambda x: x[1]))

    return [(board.san(action), value) for action, value in action_values]

if __name__ == '__main__':
    # White has M4, best move is Nf7
    #fen0 = '3q1r1k/6pp/8/3QNN2/8/8/5PPP/6K1 w - - 0 1'
    #print(ChessState(chess.Board(fen0)).rollout(1000))
    #print(eval_fen(fen0, 10000, 20))

    # White has M1, best move Rd8. Any other moves lose to M2 or M1.
    #print(eval_fen('7k/6pp/7p/7K/8/8/6q1/3R4 w - - 0 1', 1000, 10000))

    # Black has M1, best move g6#. Other moves give rough equality or worse.
    #print(eval_fen('6qk/2R4p/7K/8/8/8/8/4R3 b - - 1 1', 1000, 10000))


    # White has M2, best move Rxg8+. Any other move loses.
    print(eval_fen('2R3qk/5p1p/7K/8/8/8/5r2/2R5 w - - 0 1', 1000, 1000000))
    print(eval_fen('2R3qk/5p1p/7K/8/8/8/5r2/2R5 w - - 0 1', 1000, 1000000))
    print(eval_fen('2R3qk/5p1p/7K/8/8/8/5r2/2R5 w - - 0 1', 1000, 1000000))
    print(eval_fen('2R3qk/5p1p/7K/8/8/8/5r2/2R5 w - - 0 1', 1000, 1000000))

