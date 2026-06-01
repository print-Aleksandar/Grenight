from abc import *
from services.validator_service import *


class Piece(ABC):
    def __init__(self, uid, is_white, position):
        self.uid = uid
        self.is_white = is_white
        self.position = position
        self.had_first_move = False

    @abstractmethod
    def get_attacking_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        pass

    @abstractmethod
    def get_moving_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        pass


class Pawn(Piece):
    def __init__(self, uid, is_white, position):
        super().__init__(uid, is_white, position)
        self.is_el_passant_vulnerable = False
        self.is_next_move_pawn_promotion = is_next_move_pawn_promotion(self)

    def get_attacking_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        y, x = self.position
        sign = 1 if self.is_white else -1
        mps = [(y + sign, x - 1), (y + sign, x + 1)]

        attacking_positions = []
        for mp in mps:
            if is_position_within_board(mp):
                attacking_positions.append(mp)

        return attacking_positions

    def get_moving_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        free_positions = get_free_positions(pieces, self)

        y, x = self.position
        sign = 1 if self.is_white else -1
        mp = (y + sign, x)

        moving_positions = []
        if is_move_valid(pieces, self, mp, free_positions):
            moving_positions.append(mp)
            if not self.had_first_move:
                mp = (y + 2 * sign, x)
                if is_move_valid(pieces, self, mp, free_positions):
                    moving_positions.append(mp)

        if self.had_first_move:
            if is_en_passant_possible(pieces, self, True):
                moving_positions.append(tuple((y + sign, x - 1)))
            if is_en_passant_possible(pieces, self, False):
                moving_positions.append(tuple((y + sign, x + 1)))

        enemy_positions = get_enemy_positions(pieces, self)
        attacking_positions = [attacking_position for attacking_position in self.get_attacking_positions(pieces)
                               if is_move_valid(pieces, self, attacking_position, enemy_positions)]

        return list(set(moving_positions + attacking_positions))


def attacking_to_move_positions(pieces: list[Piece],
                                piece: Piece) \
        -> list[tuple[int, int]]:
    return [attacking_position for attacking_position in piece.get_attacking_positions(pieces)
            if is_move_valid(pieces, piece, attacking_position, get_non_ally_positions(pieces, piece))]


class Knight(Piece):
    def __init__(self, uid, is_white, position):
        super().__init__(uid, is_white, position)

    def get_attacking_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        y, x = self.position
        dyx = [(2, 1), (1, 2), (2, -1), (1, -2), (-2, 1), (-1, 2), (-2, -1), (-1, -2)]

        attacking_positions = []
        for cyx in dyx:
            cy, cx = cyx
            mp = (y + cy, x + cx)
            if is_position_within_board(mp):
                attacking_positions.append(mp)

        return attacking_positions

    def get_moving_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        return attacking_to_move_positions(pieces, self)


def get_attacking_positions_directional(pieces: list[Piece],
                                        piece: Piece,
                                        dyx: list[tuple[int, int]]) \
        -> list[tuple[int, int]]:
    free_positions = get_free_positions(pieces, piece)
    enemy_positions = get_enemy_positions(pieces, piece)
    allay_positions = get_ally_positions(pieces, piece)

    y, x = piece.position
    attacking_positions_directional = []
    for cyx in dyx:
        cy, cx = cyx

        if cy == 0:
            cx = range(x - 1, -1, -1) if cx < 0 else range(x + 1, 8, 1)
            for mx in cx:
                mp = (y, mx)
                if mp in free_positions:
                    attacking_positions_directional.append(mp)
                elif mp in enemy_positions:
                    attacking_positions_directional.append(mp)
                    break
                elif mp in allay_positions:
                    break

        elif cx == 0:
            cy = range(y - 1, -1, -1) if cy < 0 else range(y + 1, 8, 1)
            for my in cy:
                mp = (my, x)
                if mp in free_positions:
                    attacking_positions_directional.append(mp)
                elif mp in enemy_positions:
                    attacking_positions_directional.append(mp)
                    break
                elif mp in allay_positions:
                    break

        else:
            cx = range(x - 1, -1, -1) if cx < 0 else range(x + 1, 8, 1)
            cy = range(y - 1, -1, -1) if cy < 0 else range(y + 1, 8, 1)
            for mx, my in zip(cy, cx):
                mp = (my, mx)
                if not is_position_within_board(mp):
                    break

                if mp in free_positions:
                    attacking_positions_directional.append(mp)
                elif mp in enemy_positions:
                    attacking_positions_directional.append(mp)
                    break
                elif mp in allay_positions:
                    break

    return attacking_positions_directional


class Bishop(Piece):
    def __init__(self, uid, is_white, position):
        super().__init__(uid, is_white, position)

    def get_attacking_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        dyx = [(-1, -1), (1, 1), (1, -1), (-1, 1)]
        return get_attacking_positions_directional(pieces, self, dyx)

    def get_moving_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        return attacking_to_move_positions(pieces, self)


class Rook(Piece):
    def __init__(self, uid, is_white, position):
        super().__init__(uid, is_white, position)

    def get_attacking_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        dyx = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return get_attacking_positions_directional(pieces, self, dyx)

    def get_moving_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        return attacking_to_move_positions(pieces, self)


class Queen(Piece):
    def __init__(self, uid, is_white, position):
        super().__init__(uid, is_white, position)

    def get_attacking_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        dyx = [(-1, -1), (1, 1), (1, -1), (-1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
        return get_attacking_positions_directional(pieces, self, dyx)

    def get_moving_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        return attacking_to_move_positions(pieces, self)


class King(Piece):
    def __init__(self, uid, is_white, position):
        super().__init__(uid, is_white, position)

    def get_attacking_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        y, x = self.position
        dyx = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (1, -1), (-1, -1)]

        attacking_positions = []
        for cyx in dyx:
            cy, cx = cyx
            mp = (y + cy, x + cx)
            if is_position_within_board(mp):
                attacking_positions.append(mp)

        return attacking_positions

    def get_moving_positions(self, pieces: list[Piece]) \
            -> list[tuple[int, int]]:
        moving_positions = []
        if is_castling_possible(pieces, self, True):
            moving_positions.append(tuple((self.position[0], self.position[1] - 2)))
        if is_castling_possible(pieces, self, False):
            moving_positions.append(tuple((self.position[0], self.position[1] + 2)))

        return moving_positions + attacking_to_move_positions(pieces, self)