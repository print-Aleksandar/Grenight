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


class ValidMovesPieceRequest:
    def __init__(self, pieces: list[Piece],
                 uid: str,
                 is_from_white_player: bool,
                 is_white_on_turn: bool) -> None:

        self.pieces = pieces
        self.uid = uid
        self.is_from_white_player = is_from_white_player
        self.is_white_on_turn = is_white_on_turn


class AgentMoveRequest:
    def __init__(self, pieces: list[Piece],
                 is_for_white: bool,
                 is_for_white_turn: bool,
                 is_current_move_promotion: bool) -> None:

        self.pieces = pieces
        self.is_for_white = is_for_white
        self.is_for_white_turn = is_for_white_turn
        self.is_current_move_promotion = is_current_move_promotion