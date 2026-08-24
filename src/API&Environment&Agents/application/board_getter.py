from domain.pieces import Piece
from domain.board_initialization import POSITIONS


class BoardGetter:
    def __init__(self, getter_call: list) -> None:

        self.white_pieces = getter_call[0]
        self.black_pieces = getter_call[1]
        self.white_positions = getter_call[2]
        self.black_positions = getter_call[3]
        self.free_positions = getter_call[4]
        self.uid_index = getter_call[5]
        self.position_index = getter_call[6]
        self.king_color_index = getter_call[7]

    def get_piece_by_uid(self, uid: str) -> Piece | None:
        return self.uid_index.get(uid)

    def get_piece_by_position(self, position: tuple[int, int]) -> Piece | None:
        return self.position_index.get(position)

    def get_king(self, color: bool) -> Piece | None:
        return self.king_color_index.get(color)

    def is_color_king_on_position(self, color: bool, position: tuple[int, int]) -> bool:
        return self.get_king(color).position == position


def all_per_move_getter(pieces: list[Piece]) -> list:

    white_pieces = [piece for piece in pieces if piece.is_white]
    black_pieces = [piece for piece in pieces if not piece.is_white]

    white_positions = [piece.position for piece in white_pieces]
    black_positions = [piece.position for piece in black_pieces]
    free_positions = [position for position in POSITIONS
                      if position not in white_positions + black_positions]

    uid_index = {piece.uid: piece for piece in pieces}
    position_index = {piece.position: piece for piece in pieces}
    king_color_index = {piece.is_white: piece for piece in pieces if not piece.can_be_captured}

    return [white_pieces, black_pieces, white_positions,
            black_positions, free_positions,
            uid_index, position_index, king_color_index]


def get_piece_by_uid(pieces: list[Piece],
                     uid: str) -> Piece | None:
    return next((piece for piece in pieces if piece.uid == uid), None)


def get_piece_by_position(pieces: list[Piece],
                          position: tuple[int, int]) -> Piece | None:
    return next((piece for piece in pieces if piece.position == position), None)
