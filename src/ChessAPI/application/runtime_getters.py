from domain.pieces import Piece
from domain.board_configs import POSITIONS


def all_per_move_getter(pieces: list[Piece]) -> list:

    white_pieces = [piece for piece in pieces if piece.is_white]
    black_pieces = [piece for piece in pieces if not piece.is_white]

    white_positions = [piece.position for piece in white_pieces]
    black_positions = [piece.position for piece in black_pieces]
    free_positions = [position for position in POSITIONS
                      if position not in white_positions + black_positions]

    return [white_pieces, black_pieces, white_positions,
            black_positions, free_positions]


def get_piece_by_uid(pieces: list[Piece],
                     uid: str) -> Piece | None:
    return next((piece for piece in pieces if piece.uid == uid), None)


def get_piece_by_position(pieces: list[Piece],
                          position: tuple[int, int]) -> Piece | None:
    return next((piece for piece in pieces if piece.position == position), None)
