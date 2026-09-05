from collections import deque
import numpy as np


class PreviousPiecesEncodedQ:
    def __init__(self, length: int) -> None:
        self.queue: deque[np.ndarray] = deque(maxlen=length)

    def push(self, previous_pieces_encoded: np.ndarray) -> None:
        self.queue.append(previous_pieces_encoded)

    def peek(self) -> np.ndarray | None:
        return self.queue[-1] if self.queue else None

    def peek_on_idx(self, idx: int) ->  np.ndarray | None:
        try:
            return self.queue[idx]
        except IndexError:
            return None

    def rotate(self):
        for i, previous_pieces_encoded in enumerate(self.queue):

            order = np.arange(len(previous_pieces_encoded))
            offset = 4
            order[:-2] = (order[:-2] + offset) % (len(order) - 2)
            order[-2:] = order[-2:][::-1]

            self.queue[i] = previous_pieces_encoded[order, ::-1, :]

    def __len__(self) -> int:
        return len(self.queue)
