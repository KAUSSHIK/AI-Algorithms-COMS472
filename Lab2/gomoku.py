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
# utility : function that returns the utility of the state for a player (1: win, -1: loss, 0: draw)
# board   : current state of the board
# moves   : possible moves for the player to choose from

# Alpha Beta Pruning
def alpha_beta_search(state, game, depth):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)

    def max_value(state, alpha, beta, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth <= 0:
            return game.evaluate(state)
        v = float('-inf')
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth <= 0:
            return game.evaluate(state)
        v = float('inf')
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_score = float('-inf')
    beta = float('inf')
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta, depth - 1)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action

# Players for the game
def query_player(game, state):
    """Make a move by querying standard input."""
    print("current state:")
    game.display(state)
    print("")
    move = None
    while move not in game.actions(state): # Check if the move is valid
        move_string = input('Your move? ')
        try:
            move = eval(move_string)
        except:
            move = None
    return move

# Alpha Beta Player
def alpha_beta_player(game, state):
    print("Alpha-beta player thinking...")
    return alpha_beta_search(state, game, game.depth) # This agent returns the best move from the alpha beta pruning serach

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
    """Gomoku game is like an extension of TicTacToe where 5 is the sequence length to win."""
    def __init__(self, h=15, v=15, k=5, depth=3):
        TicTacToe.__init__(self, h, v, k)
        self.depth = depth
        
    def actions(self, state):
        """Legal moves."""
        moves = state.moves
        if not state.board:
            return [(self.h // 2 + 1, self.v // 2 + 1)]  # First move must be in the center (8, 8)
        elif len(state.board) == 1:
            return moves  # Second player can move anywhere
        elif len(state.board) == 2:
            # Third move (second move by the first player) must be at least two squares away from the center
            center = (self.h // 2 + 1, self.v // 2 + 1)
            return [(x, y) for x, y in moves
                    if max(abs(x - center[0]), abs(y - center[1])) >= 2]
        else:
            return moves
    
    def evaluate(self, state):
        # Use the existing compute_utility function or create a new evaluation function
        return self.compute_utility(state.board, list(state.board.keys())[-1], state.to_move)
    
    
    # Evaluation Function - 1 : Very good performance
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

    # Evaluation Function - 2 : Pretty Bad performance (mainly for comparison)
    def compute_utility2(self, board, move, player):
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
                score += (10 ** length) * open_rows
                open_rows = self.count_open_rows(board, move, other_player, length, (dx, dy)) 
                score -= (10 ** length) * open_rows
                
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
            # Display the current state of the board
            self.display(state)
            
            # Get the move from the current player
            move = players[state.to_move == 'X'](self, state)
            
            # Apply the move to the state
            state = self.result(state, move)
            
            # Check if the game is over
            if self.terminal_test(state):
                self.display(state)
                if state.utility == 0:
                    print("It's a draw!")
                else:
                    print("Player {} wins!".format('X' if state.utility > 0 else 'O'))
                return state.utility
        
    def terminal_test(self, state):
        """A state is terminal if it is won or there are no empty squares."""
        if not state.moves:
            return True

        # Check for a winning pattern around the last move
        last_move = list(state.board.keys())[-1]  # Convert dict_keys to list
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for player in ['X', 'O']:
            for dx, dy in directions:
                if self.k_in_row(state.board, last_move, player, (dx, dy)):
                    return True

        return False
        
def play_gomoku(depth):
    """Play a game of Gomoku."""
    gomoku = Gomoku(depth=depth)
    utility = gomoku.play_game(alpha_beta_player, query_player)
    return utility

def main():
    depths = [1, 2, 3, 4]
    for depth in depths:
        print("Depth:", depth)
        start_time = time.time()
        utility = play_gomoku(depth)
        end_time = time.time()
        print("Game result:", "X wins" if utility > 0 else "O wins" if utility < 0 else "Draw")
        print("Time taken:", end_time - start_time, "seconds")
        print()

if __name__ == '__main__':
    # Play a game against the AI
    depth = 2
    print("Playing a game against the AI with depth", depth)
    utility = play_gomoku(depth)
    print("Game result:", "X wins" if utility > 0 else "O wins" if utility < 0 else "Draw")