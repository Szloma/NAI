WIDTH = 4
HEIGHT = WIDTH
TOKENS_TO_WIN = 3

from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax


class Connect4(TwoPlayerGame):
    """ In each turn, a player drops a token into the chosen column,
     where it occupies the lowest available slot. The first player to align
    TOKENS_TO_WIN of their tokens horizontally, vertically, or diagonally wins the game."""

    def __init__(self, players=None):
        self.players = players
        self.board = self.generate_board()
        self.current_player = 1

    def generate_board(self):
        board = ""
        for row in range(WIDTH):
            for col in range(HEIGHT):
                board += "0"
        return board

    def show(self):
        for i in range(0, WIDTH):
            print("_", end=" ")
        print("\n")
        for i in range(len(self.board)):
            print(self.board[i], end=" ")
            if (i + 1) % WIDTH == 0:
                print()
        for i in range(0, WIDTH):
            print("_", end=" ")

    def possible_moves(self):
        moves = []
        for row in range(WIDTH):
            if self.board[row] == "0":
                moves.append(row)
        return moves

    def make_move(self, index):
        next_position = index
        stable_position = index
        for i in range(0, HEIGHT):
            if self.board[next_position] == "0":
                stable_position = next_position
                next_position += WIDTH

        if self.current_player == 1:
            token_mark = 1
        else:
            token_mark = 2
        self.board = self.board[:stable_position] + str(token_mark) + self.board[stable_position + 1:]

    def scoring(self):
        ##AI- 2 player
        winner = int(self.win())
        # if int(self.win()) == 2:
        #     return 100
        # else: return 0
        # print("WINNER:", winner)
        # print("possible moves:", self.possible_moves())
        if winner == 2:
            return 1
        elif winner == 1:
            return -10000
        else:
            return -10000

    def win(self):

        # row * width + col to check 1D as 2D (row, col)
        def idx(row, col):
            return row * WIDTH + col

        for player in (1, 2):
            token_mark = str(player)

            # horizontal
            for row in range(HEIGHT):
                for col in range(WIDTH - (TOKENS_TO_WIN - 1)):
                    if all(self.board[idx(row, col + i)] == token_mark for i in range(TOKENS_TO_WIN)):
                        return token_mark

            # vertical
            for col in range(WIDTH):
                for row in range(HEIGHT - (TOKENS_TO_WIN - 1)):
                    if all(self.board[idx(row + i, col)] == token_mark for i in range(TOKENS_TO_WIN)):
                        return token_mark

            # down‑right
            for row in range(HEIGHT - (TOKENS_TO_WIN - 1)):
                for col in range(WIDTH - (TOKENS_TO_WIN - 1)):
                    if all(self.board[idx(row + i, col + i)] == token_mark for i in range(TOKENS_TO_WIN)):
                        return token_mark

            # up‑right
            for row in range((TOKENS_TO_WIN - 1), HEIGHT):
                for col in range(WIDTH - (TOKENS_TO_WIN - 1)):
                    if all(self.board[idx(row - i, col + i)] == token_mark for i in range(TOKENS_TO_WIN)):
                        return token_mark

        return 0

    def is_over(self):
        if int(self.win()) == 2 or int(self.win()) == 1: return True
        if self.possible_moves() == []: return True
        return False


if __name__ == '__main__':
    ai = Negamax(13)
    game = Connect4([ AI_Player(ai), Human_Player()])
    history = game.play()