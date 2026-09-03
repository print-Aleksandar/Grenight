import numpy as np
from domain.pieces import PIECES_NUMBERS
from domain.configs import ROWS, COLUMNS
from environment.canonical_with_history_and_reward_shaping.pieces_queue import PiecesQueue


class PiecePlaneEncoder:

    NUM_PLANES = (10 * 4) + 3

    WHITE_PIECES_PLANE = 8
    BLACK_PIECES_PLANE = 9

    NO_PROGRESS_PLANE = NUM_PLANES - 3
    REPETITION_PLANE = NUM_PLANES - 2
    WHOSE_TURN_PLANE = NUM_PLANES - 1

    WHITE_ON_TURN_VALUE = 1.0
    BLACK_ON_TURN_VALUE = 0.0


    def encode_planes(self, pieces_q: PiecesQueue,
                      current_player_is_white: bool,
                      steps_without_progress: int = 0,
                      max_steps_without_progress: int = 1,
                      repetition_count: int = 0,
                      repetition_limit: int = 3) -> np.ndarray:

        state = np.zeros((self.NUM_PLANES, ROWS, COLUMNS), dtype=np.float32)

        state[self.WHOSE_TURN_PLANE, :, :] = (
            self.WHITE_ON_TURN_VALUE if current_player_is_white else self.BLACK_ON_TURN_VALUE
        )

        state[self.NO_PROGRESS_PLANE, :, :] = min(
            steps_without_progress / max_steps_without_progress, 1.0
        )

        state[self.REPETITION_PLANE, :, :] = min(
            repetition_count / repetition_limit, 1.0
        )

        for i in range(len(pieces_q) - 1, -1, -1):
            pieces = pieces_q.peek_on_idx(i)
            p = 3 - i
            for piece in pieces:

                base_plane = PIECES_NUMBERS[type(piece)]
                plane = base_plane if piece.is_white else base_plane + 4
                plane = plane + (p * 10)
                y, x = piece.position
                state[plane, y, x] = 1.0

                if piece.is_white:
                    state[self.WHITE_PIECES_PLANE + (10 * p), y, x] = 1.0

                else:
                    state[self.BLACK_PIECES_PLANE + (10 * p), y, x] = 1.0

        return state
