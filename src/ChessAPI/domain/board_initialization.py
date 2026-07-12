from board_configs import COLUMNS, ROWS
from pieces import Piece, Pawn, Rook, Knight, Bishop, King, Queen


ROOK_INCLUDED = COLUMNS // 2 >= 2
KNIGHT_INCLUDED = COLUMNS // 2 >= 3
BISHOP_INCLUDED = COLUMNS // 2 >= 4


def create_initial_board() -> list[Piece]:
    pieces = []

    pieces.append(Queen('wq', (0, COLUMNS // 2)))
    pieces.append(Queen('bq', (ROWS - 1, COLUMNS // 2)))

    pieces.append(King('wk', (0, (COLUMNS // 2) + 1)))
    pieces.append(King('bk', (ROWS - 1, (COLUMNS // 2) + 1)))

    pieces.extend([Pawn(('w' + str(i) + 'p'), (1, i)) for i in range(COLUMNS)])
    pieces.extend([Pawn(('b' + str(i) + 'p'), (ROWS - 2, i)) for i in range(COLUMNS)])

    if ROOK_INCLUDED:
        pieces.append(Rook('wlr', (0, 0)))
        pieces.append(Rook('wrr', (0, COLUMNS - 1)))
        pieces.append(Rook('blr', (ROWS - 1, 0)))
        pieces.append(Rook('brr', (ROWS - 1, COLUMNS - 1)))

    if KNIGHT_INCLUDED:
        pieces.append(Knight('wlk', (0, 1)))
        pieces.append(Knight('wrk', (0, COLUMNS - 2)))
        pieces.append(Knight('blk', (ROWS - 1, 1)))
        pieces.append(Knight('brk', (ROWS - 1, COLUMNS - 2)))

    if BISHOP_INCLUDED:
        pieces.append(Bishop('wlb', (0, 2)))
        pieces.append(Bishop('wrb', (0, COLUMNS - 3)))
        pieces.append(Bishop('blb', (ROWS - 1, 2)))
        pieces.append(Bishop('brb', (ROWS - 1, COLUMNS - 3)))

    return pieces


INITIAL_BOARD_STATIC = create_initial_board()


def had_first_move_func(uid: str, position: tuple[int, int]) -> bool:
    return position == next((p for p in INITIAL_BOARD_STATIC
                             if p.uid == uid), None).position


def rows_moved_func(uid: str, position: tuple[int, int]) -> int:
    return abs(position[0] - next((p for p in INITIAL_BOARD_STATIC
                                   if p.uid == uid), None).position[0])