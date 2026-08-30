from abc import ABC, abstractmethod
from domain.configs import COLUMNS, ROWS


class Piece(ABC):
    def __init__(self,
                 uid: str,
                 is_white: bool,
                 position: tuple[int, int],
                 had_first_move: bool,
                 can_be_captured: bool,
                 is_moving_sequence: bool,
                 do_attacking_position_requires_enemy_on_it: bool,
                 moving_positions: list[tuple[int, int]],
                 attacking_positions: list[tuple[int, int]]) -> None:

        # PARAMS AND DERIVED ATTRIBUTES:
        self.uid = uid
        self.is_white = is_white
        self.position = position
        self.had_first_move = had_first_move
        self.can_be_captured = can_be_captured
        self.is_moving_sequence = is_moving_sequence
        self.do_attacking_position_requires_enemy_on_it = do_attacking_position_requires_enemy_on_it
        self.moving_positions = moving_positions
        self.attacking_positions = attacking_positions

        # PAWN RUNTIME ATTRIBUTES:
        self.can_implement_pawn_moves = self.is_this_piece_pawn()
        self.is_next_move_pawn_promotion = self.is_pawn_on_pre_last_rank()
        self.is_current_move_pawn_promotion = self.is_pawn_on_last_rank()

    def is_this_piece_pawn(self) -> bool:
        return isinstance(self, Pawn)

    def is_pawn_on_pre_last_rank(self) -> bool:
        if not self.can_implement_pawn_moves:
            return False
        return ((self.is_white and self.position[0] == ROWS - 2)
                or (not self.is_white and self.position[0] == 1))

    def is_pawn_on_last_rank(self) -> bool:
        if not self.can_implement_pawn_moves:
            return False
        return ((self.is_white and self.position[0] == ROWS - 1)
                or (not self.is_white and self.position[0] == 0))

    @abstractmethod
    def update_after_flipping(self) -> None:
        pass

    @abstractmethod
    def moved_to(self, position: tuple[int, int]) -> "Piece":
        pass


def is_position_within_board(position: tuple[int, int]) -> bool:
    y, x = position
    return 0 <= y < ROWS and 0 <= x < COLUMNS


def filter_positions_within_board(positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [p for p in positions if is_position_within_board(p)]


def make_sequence(position: tuple[int, int],
                  directional: bool,
                  diagonal: bool,
                  forward_only = False,
                  forward_factor = 1,
                  max_rows_forward = 1) -> list[tuple[int, int]] | None:

    sequence = []
    y, x = position

    # PAWN SEQUENCE
    if forward_only:
        if diagonal or not directional or forward_factor not in (-1, 1) or 1 < max_rows_forward > 2:
            # Note: exception removed here, error not occurred.
            return None
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
        max_dim = max(COLUMNS, ROWS)
        sequence.extend([(y + i, x + i) for i in range(-max_dim, max_dim)
                         if 0 <= (y + i) < ROWS and 0 <= (x + i) < COLUMNS
                         and ((y + i), (x + i)) != position])
        sequence.extend([(y + i, x - i) for i in range(-max_dim, max_dim)
                         if 0 <= (y + i) < ROWS and 0 <= (x - i) < COLUMNS
                         and ((y + i), (x - i)) != position])

    return filter_positions_within_board(sequence)


class Pawn(Piece):
    def __init__(self,
                 uid: str,
                 is_white: bool,
                 position: tuple[int, int],
                 had_first_move: bool) -> None:

        y, x = position

        forward_factor = 1 if is_white else -1
        moving_positions = filter_positions_within_board(make_sequence(
            position=position,
            directional=True,
            diagonal=False,
            forward_only=True,
            forward_factor=forward_factor,
            max_rows_forward=1))

        attacking_positions = filter_positions_within_board(
            [(y + (1 * forward_factor), x + 1), (y + (1 * forward_factor), x - 1)]
        )

        super().__init__(uid=uid,
                         is_white=is_white,
                         position=position,
                         had_first_move=had_first_move,
                         can_be_captured=True,
                         is_moving_sequence=True,
                         do_attacking_position_requires_enemy_on_it=True,
                         moving_positions=moving_positions,
                         attacking_positions=attacking_positions)

    def update_after_flipping(self) -> None:
        y, x = self.position

        forward_factor = 1 if self.is_white else -1
        self.moving_positions = filter_positions_within_board(make_sequence(
            position=self.position,
            directional=True,
            diagonal=False,
            forward_only=True,
            forward_factor=forward_factor,
            max_rows_forward=1))

        self.attacking_positions = filter_positions_within_board(
            [(y + (1 * forward_factor), x + 1), (y + (1 * forward_factor), x - 1)]
        )

        self.is_next_move_pawn_promotion = self.is_pawn_on_pre_last_rank()
        self.is_current_move_pawn_promotion = self.is_pawn_on_last_rank()

    def moved_to(self, position: tuple[int, int]) -> "Piece":
        return Pawn(uid=self.uid, is_white=self.is_white,
                    position=position, had_first_move=True)


class Knight(Piece):
    def __init__(self,
                 uid: str,
                 is_white: bool,
                 position: tuple[int, int],
                 had_first_move: bool) -> None:

        y, x = position

        dyx = [(2, 1), (1, 2), (2, -1), (1, -2), (-2, 1), (-1, 2), (-2, -1), (-1, -2)]
        positions = filter_positions_within_board([(y + dy, x + dx) for (dy, dx) in dyx])

        super().__init__(uid=uid,
                         is_white=is_white,
                         position=position,
                         had_first_move=had_first_move,
                         can_be_captured=True,
                         is_moving_sequence=False,
                         do_attacking_position_requires_enemy_on_it=False,
                         moving_positions=positions,
                         attacking_positions=positions)

    def update_after_flipping(self) -> None:
        y, x = self.position

        dyx = [(2, 1), (1, 2), (2, -1), (1, -2), (-2, 1), (-1, 2), (-2, -1), (-1, -2)]
        positions = filter_positions_within_board([(y + dy, x + dx) for (dy, dx) in dyx])

        self.attacking_positions = positions
        self.moving_positions = positions

    def moved_to(self, position: tuple[int, int]) -> "Piece":
        return Knight(uid=self.uid, is_white=self.is_white,
                      position=position, had_first_move=True)


class Bishop(Piece):
    def __init__(self, uid: str,
                 is_white: bool,
                 position: tuple[int, int],
                 had_first_move: bool) -> None:

        positions = filter_positions_within_board(make_sequence(
            position=position,
            directional=False,
            diagonal=True,
            forward_only=False
        ))

        super().__init__(uid=uid,
                         is_white=is_white,
                         position=position,
                         had_first_move=had_first_move,
                         can_be_captured=True,
                         is_moving_sequence=True,
                         do_attacking_position_requires_enemy_on_it=False,
                         moving_positions=positions,
                         attacking_positions=positions)

    def update_after_flipping(self) -> None:
        positions = filter_positions_within_board(make_sequence(
            position=self.position,
            directional=False,
            diagonal=True,
            forward_only=False
        ))

        self.attacking_positions = positions
        self.moving_positions = positions

    def moved_to(self, position: tuple[int, int]) -> "Piece":
        return Bishop(uid=self.uid, is_white=self.is_white,
                      position=position, had_first_move=True)


class Rook(Piece):
    def __init__(self, uid: str,
                 is_white: bool,
                 position: tuple[int, int],
                 had_first_move: bool) -> None:

        positions = filter_positions_within_board(make_sequence(
            position=position,
            directional=True,
            diagonal=False,
            forward_only=False
        ))

        super().__init__(uid=uid,
                         is_white=is_white,
                         position=position,
                         had_first_move=had_first_move,
                         can_be_captured=True,
                         is_moving_sequence=True,
                         do_attacking_position_requires_enemy_on_it=False,
                         moving_positions=positions,
                         attacking_positions=positions)

    def update_after_flipping(self) -> None:
        positions = filter_positions_within_board(make_sequence(
            position=self.position,
            directional=True,
            diagonal=False,
            forward_only=False
        ))

        self.attacking_positions = positions
        self.moving_positions = positions

    def moved_to(self, position: tuple[int, int]) -> "Piece":
        return Rook(uid=self.uid, is_white=self.is_white,
                    position=position, had_first_move=True)


class Queen(Piece):
    def __init__(self, uid: str,
                 is_white: bool,
                 position: tuple[int, int],
                 had_first_move: bool) -> None:

        positions = filter_positions_within_board(make_sequence(
            position=position,
            directional=True,
            diagonal=True,
            forward_only=False
        ))

        super().__init__(uid=uid,
                         is_white=is_white,
                         position=position,
                         had_first_move=had_first_move,
                         can_be_captured=True,
                         is_moving_sequence=True,
                         do_attacking_position_requires_enemy_on_it=False,
                         moving_positions=positions,
                         attacking_positions=positions)

    def update_after_flipping(self) -> None:
        positions = filter_positions_within_board(make_sequence(
            position=self.position,
            directional=True,
            diagonal=True,
            forward_only=False
        ))

        self.attacking_positions = positions
        self.moving_positions = positions

    def moved_to(self, position: tuple[int, int]) -> "Piece":
        return Queen(uid=self.uid, is_white=self.is_white,
                     position=position, had_first_move=True)


class King(Piece):
    def __init__(self, uid: str,
                 is_white: bool,
                 position: tuple[int, int],
                 had_first_move: bool) -> None:

        y, x = position

        dyx = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        positions = filter_positions_within_board([(y + dy, x + dx) for (dy, dx) in dyx])

        super().__init__(uid=uid,
                         is_white=is_white,
                         position=position,
                         had_first_move=had_first_move,
                         can_be_captured=False,
                         is_moving_sequence=False,
                         do_attacking_position_requires_enemy_on_it=False,
                         moving_positions=positions,
                         attacking_positions=positions)

    def update_after_flipping(self) -> None:
        y, x = self.position

        dyx = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
        positions = filter_positions_within_board([(y + dy, x + dx) for (dy, dx) in dyx])

        self.attacking_positions = positions
        self.moving_positions = positions

    def moved_to(self, position: tuple[int, int]) -> "Piece":
        return King(uid=self.uid, is_white=self.is_white,
                    position=position, had_first_move=True)


PIECES_CLASSES = {
    0: Pawn,
    1: Knight,
    2: Bishop,
    3: Rook,
    4: Queen,
    5: King
}

PIECES_NUMBERS = dict()
for num, cl in PIECES_CLASSES.items():
    PIECES_NUMBERS[cl] = num
