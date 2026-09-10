from domain.pieces import Piece


class MoveResponse:
    def __init__(self, pieces: list[Piece],
                 is_game_finished: bool,
                 is_draw: bool,
                 is_white_winner: bool,
                 is_white_on_turn: bool,
                 is_next_move_promotion: bool,
                 is_enemy_in_check: bool | None = None) -> None:

        self.pieces = pieces
        self.is_game_finished = is_game_finished
        self.is_draw = is_draw
        self.is_white_winner = is_white_winner
        self.is_white_on_turn = is_white_on_turn
        self.is_next_move_promotion = is_next_move_promotion
        self.is_enemy_in_check = is_enemy_in_check


class ValidMovesPieceResponse:
    def __init__(self, uid: str,
                 valid_moves: list[tuple[int, int]] | None) -> None:

        self.uid = uid
        self.valid_moves = valid_moves
