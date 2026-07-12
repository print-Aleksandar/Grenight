from abc import ABC
from board_configs import COLUMNS, ROWS, PAWN_DOUBLE_STEP
from board_initialization import had_first_move_func, rows_moved_func
from domain.exceptions import ForwardOnlyException


class Piece(ABC):
    def __init__(self,
                 uid: str,
                 position: tuple[int, int],
                 can_be_captured: bool,
                 had_first_move: bool,
                 is_moving_sequence: bool,
                 do_attacking_position_requires_enemy_on_it: bool,
                 can_be_on_attacked_position: bool,
                 moving_positions: list[tuple[int, int]],
                 attacking_positions: list[tuple[int, int]]) -> None:

        # PARAMS AND DERIVED ATTRIBUTES:
        self.uid = uid
        self.is_white = uid[0] == 'w'
        self.position = position
        self.can_be_captured = can_be_captured
        self.had_first_move = had_first_move
        self.is_moving_sequence = is_moving_sequence
        self.do_attacking_position_requires_enemy_on_it = do_attacking_position_requires_enemy_on_it
        self.can_be_on_attacked_position = can_be_on_attacked_position
        self.moving_positions = moving_positions
        self.attacking_positions = attacking_positions

        # PAWN RUNTIME ATTRIBUTES:
        rows_moved = rows_moved_func(uid=uid, position=position)

        self.can_implement_pawn_moves = len(uid) > 2 and uid[2] == 'p'
        self.is_next_move_pawn_promotion = self.can_implement_pawn_moves and rows_moved == ROWS - 3

        self.is_en_passant_vulnerable = False # IMPLEMENTED IN APPLICATION LAYER
        self.can_capture_en_passant = PAWN_DOUBLE_STEP and rows_moved == 3


def is_position_within_board(position: tuple[int, int]) -> bool:
    y, x = position
    return 0 <= y < ROWS and 0 <= x < COLUMNS


def filter_positions_within_board(positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [p for p in positions if is_position_within_board(p)]


def forward_factor_by_uid(uid: str) -> int:
    return 1 if uid[0] == 'w' else -1


def make_sequence(position: tuple[int, int],
                  directional: bool,
                  diagonal: bool,
                  forward_only = False,
                  forward_factor = 1,
                  max_rows_forward = 1) -> list[tuple[int, int]]:

    sequence = []
    y, x = position

    # PAWN SEQUENCE
    if forward_only:
        if diagonal or not directional or forward_factor not in (-1, 1) or 1 < max_rows_forward > 2:
            raise ForwardOnlyException()
        sequence.append((y + (1 * forward_factor), x))
        if max_rows_forward == 2:
            sequence.append((y + (2 * forward_factor), x))

        return filter_positions_within_board(sequence)

    # ROOK/QUEEN SEQUENCE
    if directional:
        sequence.extend([(i, x) for i in range(0, ROWS) if i != y])
        sequence.extend([(y, i) for i in range(0, COLUMNS) if i != x])

    # BISHOP/QUEEN SEQUENCE
    if diagonal:
        min_dim = max(COLUMNS, ROWS)
        sequence.extend([(y + i, x + i) for i in range(-min_dim, min_dim)
                         if 0 <= (y + i) < ROWS and 0 <= (x + i) < COLUMNS
                         and ((y + i), (x + i)) != position])
        sequence.extend([(y + i, x - i) for i in range(-min_dim, min_dim)
                         if 0 <= (y + i) < ROWS and 0 <= (x - i) < COLUMNS
                         and ((y + i), (x - i)) != position])

    return filter_positions_within_board(sequence)


class Pawn(Piece):
    def __init__(self,
                 uid: str,
                 position: tuple[int, int]) -> None:

        y, x = position
        had_first_move = had_first_move_func(uid=uid, position=position)

        forward_factor = forward_factor_by_uid(uid=uid)
        moving_positions = filter_positions_within_board(make_sequence(
                             position=position,
                             directional=True,
                             diagonal=False,
                             forward_only=True,
                             forward_factor=forward_factor,
                             max_rows_forward=2 if had_first_move else 1))

        attacking_positions = filter_positions_within_board(
            [(y + (1 * forward_factor), x + 1), (y + (1 * forward_factor), x - 1)]
        )

        super().__init__(uid=uid,
                         position=position,
                         can_be_captured=True,
                         had_first_move=had_first_move,
                         is_moving_sequence=True,
                         do_attacking_position_requires_enemy_on_it=True,
                         can_be_on_attacked_position=True,
                         moving_positions=moving_positions,
                         attacking_positions=attacking_positions)


class Knight(Piece):
    def __init__(self,
                 uid: str,
                 position: tuple[int, int]) -> None:

        y, x = position
        had_first_move = had_first_move_func(uid=uid, position=position)

        dyx = [(2, 1), (1, 2), (2, -1), (1, -2), (-2, 1), (-1, 2), (-2, -1), (-1, -2)]
        positions = filter_positions_within_board([(y + dy, x + dx) for (dy, dx) in dyx])

        super().__init__(uid=uid,
                         position=position,
                         can_be_captured=True,
                         had_first_move=had_first_move,
                         is_moving_sequence=False,
                         do_attacking_position_requires_enemy_on_it=False,
                         can_be_on_attacked_position=True,
                         moving_positions=positions,
                         attacking_positions=positions)


class Bishop(Piece):
    def __init__(self, uid: str,
                 position: tuple[int, int]) -> None:

        had_first_move = had_first_move_func(uid=uid, position=position)

        positions = filter_positions_within_board(make_sequence(
            position=position,
            directional=False,
            diagonal=True,
            forward_only=False
        ))

        super().__init__(uid=uid,
                         position=position,
                         can_be_captured=True,
                         had_first_move=had_first_move,
                         is_moving_sequence=True,
                         do_attacking_position_requires_enemy_on_it=True,
                         can_be_on_attacked_position=True,
                         moving_positions=positions,
                         attacking_positions=positions)


class Rook(Piece):
    def __init__(self, uid: str,
                 position: tuple[int, int]) -> None:

        had_first_move = had_first_move_func(uid=uid, position=position)

        positions = filter_positions_within_board(make_sequence(
            position=position,
            directional=True,
            diagonal=False,
            forward_only=False
        ))

        super().__init__(uid=uid,
                         position=position,
                         can_be_captured=True,
                         had_first_move=had_first_move,
                         is_moving_sequence=True,
                         do_attacking_position_requires_enemy_on_it=True,
                         can_be_on_attacked_position=True,
                         moving_positions=positions,
                         attacking_positions=positions)


class Queen(Piece):
    def __init__(self, uid: str,
                 position: tuple[int, int]) -> None:

        had_first_move = had_first_move_func(uid=uid, position=position)

        positions = filter_positions_within_board(make_sequence(
            position=position,
            directional=True,
            diagonal=True,
            forward_only=False
        ))

        super().__init__(uid=uid,
                         position=position,
                         can_be_captured=True,
                         had_first_move=had_first_move,
                         is_moving_sequence=True,
                         do_attacking_position_requires_enemy_on_it=True,
                         can_be_on_attacked_position=True,
                         moving_positions=positions,
                         attacking_positions=positions)


class King(Piece):
    def __init__(self, uid: str,
                 position: tuple[int, int]) -> None:

        y, x = position
        had_first_move = had_first_move_func(uid=uid, position=position)

        dyx = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        positions = filter_positions_within_board([(y + dy, x + dx) for (dy, dx) in dyx])

        super().__init__(uid=uid,
                         position=position,
                         can_be_captured=False,
                         had_first_move=had_first_move,
                         is_moving_sequence=False,
                         do_attacking_position_requires_enemy_on_it=False,
                         can_be_on_attacked_position=False,
                         moving_positions=positions,
                         attacking_positions=positions)
