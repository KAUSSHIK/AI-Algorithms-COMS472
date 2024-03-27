"""
COMS 472 : AI - Lab 2
Author: Kausshik Manojkumar
Ciatations:
- AIMA Python Code : https://github.com/aimacode/aima-python/GAME.py
"""

from collections import namedtuple
from copy import deepcopy
import random
import time
import numpy as np

infinity = float('inf')

GameState = namedtuple('GameState', 'to_move, utility, board, moves') # Game state
# to_move : player to move
# utility :
# board : current state of the board
# moves : possible moves for the player to choose from

# Alpha Beta Pruning
def alpha_beta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""
    # game : instance of Game class
    # state : current state of the game

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

# Players for the game

def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("available moves: {}".format(game.actions(state)))
    print("")
    move = None
    if game.actions(state):
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except NameError:
            move = move_string
    else:
        print('no legal moves: passing turn to next player')
    return move

# Alpha Beta Player
def alpha_beta_player(game, state):
    return alpha_beta_search(state, game) # This agent returns the best move from the alpha beta pruning serach

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))


class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""

    def __init__(self, h=3, v=3, k=3):
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1)
                 for y in range(1, v + 1)]
        self.initial = GameState(to_move='X', utility=0, board={}, moves=moves)

    def actions(self, state):
        """Legal moves are any square not yet taken."""
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(to_move=('O' if state.to_move == 'X' else 'X'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == 'X' else -state.utility

    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), '.'), end=' ')
            print()

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.k_in_row(board, move, player, (0, 1)) or
                self.k_in_row(board, move, player, (1, 0)) or
                self.k_in_row(board, move, player, (1, -1)) or
                self.k_in_row(board, move, player, (1, 1))):
            return +1 if player == 'X' else -1
        else:
            return 0

    def k_in_row(self, board, move, player, delta_x_y):
        """Return true if there is a line through move on board for player."""
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k
    
class Gomoku(TicTacToe):
    """Gomoku game."""
    def __init__(self, h=15, v=15, k=5, depth=3):
        TicTacToe.__init__(self, h, v, k)
        self.depth = depth
        
    def actions(self, state):
        """Legal moves."""
        moves = state.moves
        if state.board.get((self.h // 2 + 1, self.v // 2 + 1)) is None:
            return [(self.h // 2 + 1, self.v // 2 + 1)]  # Black's first move must be in the center
        if len(state.board) == 1:
            return moves  # White's first move can be anywhere
        if len(state.board) == 2:
            center = (self.h // 2 + 1, self.v // 2 + 1)
            first_move = next(iter(state.board))
            diff_x = abs(first_move[0] - center[0])
            diff_y = abs(first_move[1] - center[1])
            if diff_x < 3 and diff_y < 3:
                return [(x, y) for x, y in moves 
                        if abs(x - center[0]) >= 3 or abs(y - center[1]) >= 3]  # Black's second move must be at least 3 spaces away from the center
        return moves
        
    def compute_utility(self, board, move, player):
        """Compute the utility of a board state."""
        score = 0
        other_player = 'X' if player == 'O' else 'O'

        # Check for 5-in-a-row
        if self.k_in_row(board, move, player, (0, 1)) or \
           self.k_in_row(board, move, player, (1, 0)) or \
           self.k_in_row(board, move, player, (1, 1)) or \
           self.k_in_row(board, move, player, (1, -1)):
            return np.inf if player == 'X' else -np.inf
        
        # Count rows of 2, 3, and 4 stones
        for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            for length in [2, 3, 4]:
                open_rows = self.count_open_rows(board, move, player, length, (dx, dy))
                semi_open_rows = self.count_semiopen_rows(board, move, player, length, (dx, dy))
                score += (10 ** length) * open_rows + (10 ** (length - 1)) * semi_open_rows
                open_rows = self.count_open_rows(board, move, other_player, length, (dx, dy)) 
                semi_open_rows = self.count_semiopen_rows(board, move, other_player, length, (dx, dy))
                score -= (10 ** length) * open_rows + (10 ** (length - 1)) * semi_open_rows
                
        # Add a bonus for occupying the center
        center = (self.h // 2 + 1, self.v // 2 + 1)
        if board.get(center) == player:
            score += 10
        elif board.get(center) == other_player:
            score -= 10
            
        return score
    
    def count_open_rows(self, board, move, player, length, direction):
        """Count the number of open rows of a given length in a given direction."""
        count = 0
        x, y = move
        dx, dy = direction
        for i in range(1, length + 1):
            if board.get((x + i * dx, y + i * dy)) == player:
                count += 1
            else:
                break
        for i in range(1, length + 1):
            if board.get((x - i * dx, y - i * dy)) == player:
                count += 1
            else:
                break
        return 1 if count >= length else 0
    
    def count_semiopen_rows(self, board, move, player, length, direction):
        """Count the number of semi-open rows of a given length in a given direction."""
        count = 0
        x, y = move
        dx, dy = direction
        blocked = False
        for i in range(1, length + 1):
            if board.get((x + i * dx, y + i * dy)) == player:
                count += 1
            elif board.get((x + i * dx, y + i * dy)) is not None:
                blocked = True
                break
        if not blocked:
            for i in range(1, length + 1):
                if board.get((x - i * dx, y - i * dy)) == player:
                    count += 1
                elif board.get((x - i * dx, y - i * dy)) is not None:
                    break
        return 1 if count >= length else 0
        
    def play_game(self, *players):
        """Play a game of Gomoku."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    if state.utility == 0:
                        print("It's a draw!")
                    else:
                        print("Player {} wins!".format('X' if state.utility > 0 else 'O'))
                    return state.utility
        
def play_gomoku(depth):
    """Play a game of Gomoku."""
    gomoku = Gomoku(depth=depth)
    utility = gomoku.play_game(alpha_beta_player, query_player)
    return utility

if __name__ == '__main__':
    # Compare different search depths
    depths = [1, 2, 3, 4]
    for depth in depths:
        print("Depth:", depth)
        start_time = time.time()
        utility = play_gomoku(depth)
        end_time = time.time()
        print("Game result:", "X wins" if utility > 0 else "O wins" if utility < 0 else "Draw")
        print("Time taken:", end_time - start_time, "seconds")
        print()