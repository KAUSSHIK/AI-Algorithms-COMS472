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

    """Initial state of the game - Gomoku
    - 15x15 board
    - 2 players (Black and White)
    - Black starts first
    - 5 in a row wins
    - 2D array to represent the board
    - 0 : Empty
    - 1 : Black
    - 2 : White
    - QUIRKY REQUIREMENT: The first player's first stone must be placed
        in the center of the board. The second player's first stone may be placed anywhere on the board.
        The first player's second stone must be placed at least three intersections away from the first
        stone (two empty intersections in between the two stones). Other moves are normal and only can be made on empty intersections by the player currently playing the game."""
    
    def __init__(self, n=15):
        self.n = n
        self.initial = GameState(to_move=1, utility=0, board=[[0 for _ in range(n)] for _ in range(n)], moves=[(7, 7)])
        self.initial.board[7][7] = 1
        self.initial.moves = [(i, j) for i in range(n) for j in range(n) if self.initial.board[i][j] == 0]

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        if self.terminal_test(state):
            return []
        else:
            return state.moves

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        if move not in state.moves:
            return state
        board = deepcopy(state.board)
        board[move[0]][move[1]] = state.to_move
        moves = deepcopy(state.moves)
        moves.remove(move)
        return GameState(to_move=3 - state.to_move, utility=self.compute_utility(board, move, state.to_move), board=board, moves=moves)

    def utility(self, state, player):
        """Return the value of this final state to player."""
        return state.utility if player == 1 else -state.utility

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return state.utility != 0 or len(state.moves) == 0

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        board = state.board
        print("Player to move: ", state.to_move)
        print("Utility: ", state.utility)
        for row in board:
            print(" ".join([[".", "1", "2"][cell] for cell in row]))
        print("")

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    # def play_game(self, *players):
    #     """Play an n-person, move-alternating game."""
    #     state = self.initial
    #     while True:
    #         for player in players:
    #             move = player(self, state)
    #             state = self.result(state, move)
    #             if self.terminal_test(state):
    #                 self.display(state)
    #                 return self.utility(state, self.to_move(self.initial))

    def compute_utility(self, board, move, player):
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (self.count_stones(board, move, (0, 1)) >= 5 or
            self.count_stones(board, move, (1, 0)) >= 5 or
            self.count_stones(board, move, (1, -1)) >= 5 or
            self.count_stones(board, move, (1, 1)) >= 5):
            return 1 if player == 1 else -1
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
