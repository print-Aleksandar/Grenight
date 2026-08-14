from domain.pieces import Piece


class MoveRequest:
    def __init__(self, pieces: list[Piece],
                 uid: str,
                 position: tuple[int, int],
                 is_white_on_turn: bool,
                 is_from_white_player: bool,
                 is_current_move_promotion: bool,
                 promote_to: int | None) -> None:

        self.pieces = pieces
        self.uid = uid
        self.position = position
        self.is_from_white_player = is_from_white_player
        self.is_white_on_turn = is_white_on_turn
        self.is_current_move_promotion = is_current_move_promotion
        self.promote_to = promote_to


class ValidMovesPlayerRequest:
    def __init__(self, pieces: list[Piece],
                 is_from_white_player: bool,
                 is_white: bool,
                 is_white_on_turn: bool) -> None:

        self.pieces = pieces
        self.is_white = is_white
        self.is_from_white_player = is_from_white_player
        self.is_white_on_turn = is_white_on_turn


class ValidMovesPieceRequest:
    def __init__(self, pieces: list[Piece],
                 uid: str,
                 is_from_white_player: bool,
                 is_white_on_turn: bool) -> None:

        self.pieces = pieces
        self.uid = uid
        self.is_from_white_player = is_from_white_player
        self.is_white_on_turn = is_white_on_turn
