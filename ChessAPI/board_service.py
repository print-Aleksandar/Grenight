from pieces import Piece
from static import POSITIONS


def get_white_pieces(pieces: list[Piece]) \
        -> list[Piece]:
    return [piece for piece in pieces if piece.is_white]


def get_black_pieces(pieces: list[Piece]) \
        -> list[Piece]:
    return [piece for piece in pieces if not piece.is_white]


def get_ally_pieces(pieces: list[Piece],
                     piece: Piece) \
        -> list[Piece]:
    if piece.is_white:
        return get_white_pieces(pieces)
    else:
        return get_black_pieces(pieces)


def get_enemy_pieces(pieces: list[Piece],
                     piece: Piece) \
        -> list[Piece]:
    if piece.is_white:
        return get_black_pieces(pieces)
    else:
        return get_white_pieces(pieces)


def get_ally_positions(pieces: list[Piece],
                        piece: Piece) \
        -> list[tuple[int, int]]:
    return [piece.position for piece in get_ally_pieces(pieces, piece)]


def get_enemy_positions(pieces: list[Piece],
                        piece: Piece) \
        -> list[tuple[int, int]]:
    return [piece.position for piece in get_enemy_pieces(pieces, piece)]


def get_non_ally_positions(pieces: list[Piece],
                            piece: Piece) \
        -> list[tuple[int, int]]:
    positions_of_allay_pieces = get_ally_positions(pieces, piece)
    return [position for position in POSITIONS if
            position not in positions_of_allay_pieces]


def get_free_positions(pieces: list[Piece],
                       piece: Piece) \
        -> list[tuple[int, int]]:
    positions_of_allay_pieces = get_ally_positions(pieces, piece)
    positions_of_enemy_pieces = get_enemy_positions(pieces, piece)
    return [position for position in POSITIONS if
            position not in positions_of_allay_pieces and
            position not in positions_of_enemy_pieces]


def get_piece_by_uid(pieces: list[Piece],
                     uid: str) \
        -> Piece | None:
    for piece in pieces:
        if piece.uid == uid:
            return piece
    return None


def get_piece_by_position(pieces: list[Piece],
                          position: tuple[int, int]) \
        -> Piece | None:
    for piece in pieces:
        if piece.position == position:
            return piece
    return None


def get_ally_king(pieces: list[Piece],
                   piece: Piece) \
        -> Piece:
    uid = ('w' if piece.is_white else 'b') + 'k'
    return get_piece_by_uid(pieces, uid)


def get_enemy_king(pieces: list[Piece],
                   piece: Piece) \
        -> Piece:
    uid = ('w' if not piece.is_white else 'b') + 'k'
    return get_piece_by_uid(pieces, uid)


def get_enemy_pawns(pieces: list[Piece],
                    piece: Piece) \
        -> list[Piece]:
    return [piece for piece in get_enemy_pieces(pieces, piece) \
            if piece.uid[1] == 'p']


def get_ally_attacking_positions(pieces: list[Piece],
                                  piece: Piece) \
        -> list[tuple[int, int]]:
    attacking_positions = set()
    for allay_piece in get_ally_pieces(pieces, piece):
        attacking_positions.update(allay_piece.get_attacking_positions(pieces))

    return list(attacking_positions)


def get_enemy_attacking_positions(pieces: list[Piece],
                                  piece: Piece) \
        -> list[tuple[int, int]]:
    attacking_positions = set()
    for enemy_piece in get_enemy_pieces(pieces, piece):
        attacking_positions.update(enemy_piece.get_attacking_positions(pieces))

    return list(attacking_positions)


def get_enemy_valid_moving_positions(pieces: list[Piece],
                                     piece: Piece) \
        -> list[tuple[int, int]]:
    valid_moving_positions = set()
    for enemy_piece in get_enemy_pieces(pieces, piece):
        valid_moving_positions.update(enemy_piece.get_moving_positions(pieces))

    return list(valid_moving_positions)