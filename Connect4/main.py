
WIDTH = 7
HEIGHT = 6
class Connect4:
    def __init__(self, players=None):
        self.players=players
        self.board= self.generate_board()
        self.current_player = 1
    def generate_board(self):
        board = ""
        for row in range(WIDTH):
            for col in range(HEIGHT):
                board += "0"
        return board

    def show(self):
        for i in range(0,WIDTH):
            print("_",end=" ")
        print("\n")
        for i in range(len(self.board)):
            print(self.board[i], end=" ")
            if (i + 1) % WIDTH  == 0:
                print()
        for i in range(0,WIDTH):
            print("_",end=" ")

    def possible_moves(self):
        moves = []
        for row in range(WIDTH):
            if self.board[row] == "0":
                moves.append(row)
        return moves

    def make_move(self,index, player):

        next_position = index
        stable_position = index
        for i in range(0,HEIGHT):
            if self.board[next_position]=="0":
                stable_position=next_position
                next_position+=WIDTH


        self.board = self.board[:stable_position] + str(player) + self.board[stable_position + 1:]

    def win(self):
        ##
        lengths = {"1":{},"2":{}}

        start_indexes=[]
        for i in range(WIDTH):
            start_indexes.append(i)
            for key in lengths.keys():
                lengths[key][i]=0

        ##Vertical check
        for i in start_indexes:
            next_index=i
            for n in range(HEIGHT):
                next_index+=WIDTH
                if next_index >= len(self.board):
                    break
                if self.board[next_index]!="0":
                    lengths[self.board[next_index]][i]+=1
                    print("match at ", n, " ",i)
        ##TODO diagonal check

        ##Determine if winner
        for key in lengths.keys():
            value = lengths[key]
            for n in value.keys():
                if n==4:
                    return key
        return 0

    def scoring(self):
        ##AI - 2 player
        if self.win()==2:
            return 100
        return 0

if __name__ == '__main__':
    app = Connect4()
    print(app.possible_moves())
    app.show()
    app.make_move(1,1)
    app.make_move(1, 1)
    app.make_move(1, 1)
    app.make_move(1, 2)
    app.show()
    print(app.win())

