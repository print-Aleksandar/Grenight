from domain.configs import COLUMNS, ROWS
from domain.pieces import Piece, Pawn, Rook, Knight, Bishop, King, Queen


POSITIONS = []
[POSITIONS.extend([(y,x) for x in range(COLUMNS)]) for y in range(ROWS)]


def create_initial_board() -> list[Piece]:
    pieces = []

    pieces.append(Queen('wq',True, (0, (COLUMNS // 2) - 1), False))
    pieces.append(Queen('bq',False, (ROWS - 1, (COLUMNS // 2) - 1), False))

    pieces.append(King('wk',True, (0, COLUMNS // 2), False))
    pieces.append(King('bk',False, (ROWS - 1, COLUMNS // 2), False))

    pieces.extend([Pawn(('w' + str(i) + 'p'),True,(1, i), False) for i in range(COLUMNS)])
    pieces.extend([Pawn(('b' + str(i) + 'p'),False,(ROWS - 2, i), False) for i in range(COLUMNS)])

    pieces.append(Rook('wlr', True, (0, 0), False))
    pieces.append(Rook('wrr', True, (0, COLUMNS - 1), False))
    pieces.append(Rook('blr', False, (ROWS - 1, 0), False))
    pieces.append(Rook('brr', False, (ROWS - 1, COLUMNS - 1), False))

    if COLUMNS // 2 == 3:
        pieces.append(Knight('wlk', True, (0, 1), False))
        pieces.append(Knight('wrk', True, (0, COLUMNS - 2), False))
        pieces.append(Knight('blk', False, (ROWS - 1, 1), False))
        pieces.append(Knight('brk', False, (ROWS - 1, COLUMNS - 2), False))

    if COLUMNS // 2 == 4:
        pieces.append(Bishop('wlb', True, (0, 2), False))
        pieces.append(Bishop('wrb', True, (0, COLUMNS - 3), False))
        pieces.append(Bishop('blb', False, (ROWS - 1, 2), False))
        pieces.append(Bishop('brb', False, (ROWS - 1, COLUMNS - 3), False))

    return pieces
