import numpy as np
from domain.pieces import Piece, Pawn, Rook, Knight, Bishop,Queen, King
from domain.board_configs import ROWS, COLUMNS


class PiecePlaneEncoder:

    NUM_PLANES = 14
    SIDE_TO_MOVE_PLANE = 12
    PROMOTION_PLANE = 13

    PLANES = {
        (True, Pawn): 0,
        (True, Rook): 1,
        (True, Knight): 2,
        (True, Bishop): 3,
        (True, Queen): 4,
        (True, King): 5,

        (False, Pawn): 6,
        (False, Rook): 7,
        (False, Knight): 8,
        (False, Bishop): 9,
        (False, Queen): 10,
        (False, King): 11,
    }

    def encode(self, pieces: list[Piece],
               is_white_on_turn: bool,
               is_promotion: bool) -> np.ndarray:

        state = np.zeros(
            (self.NUM_PLANES, ROWS, COLUMNS),
            dtype=np.float32
        )

        for piece in pieces:
            plane = self._get_plane(piece)
            y, x = piece.position

            state[plane, y, x] = 1.0

        if is_white_on_turn:
            state[self.SIDE_TO_MOVE_PLANE, :, :] = 1.0

        if is_promotion:
            state[self.PROMOTION_PLANE, :, :] = 1.0

        return state

    def _get_plane(self, piece: Piece) -> int:

        for (is_white, piece_type), plane in self.PLANES.items():
            if piece.is_white == is_white and isinstance(piece, piece_type):
                return plane

        raise ValueError(
            f"Unsupported piece type: {type(piece).__name__}"
        )