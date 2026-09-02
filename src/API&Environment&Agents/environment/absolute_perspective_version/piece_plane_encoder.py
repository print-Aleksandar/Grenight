import numpy as np
from domain.pieces import Piece, PIECES_NUMBERS
from domain.configs import ROWS, COLUMNS


class PiecePlaneEncoder:

    NUM_PLANES = 12

    WHITE_PIECES_PLANE = 8
    BLACK_PIECES_PLANE = 9
    NO_PROGRESS_PLANE = 10
    REPETITION_PLANE = 11


    def encode_planes(self, pieces: list[Piece],
                      steps_without_progress: int = 0,
                      max_steps_without_progress: int = 1,
                      repetition_count: int = 0,
                      repetition_limit: int = 3) -> np.ndarray:

        state = np.zeros((self.NUM_PLANES, ROWS, COLUMNS), dtype=np.float32)

        state[self.NO_PROGRESS_PLANE, :, :] = min(
            steps_without_progress / max_steps_without_progress, 1.0
        )

        state[self.REPETITION_PLANE, :, :] = min(
            repetition_count / repetition_limit, 1.0
        )

        for piece in pieces:

            base_plane = PIECES_NUMBERS[type(piece)]
            plane = base_plane if piece.is_white else base_plane + 4
            y, x = piece.position
            state[plane, y, x] = 1.0

            if piece.is_white:
                state[self.WHITE_PIECES_PLANE, y, x] = 1.0

            else:
                state[self.BLACK_PIECES_PLANE, y, x] = 1.0

        return state
