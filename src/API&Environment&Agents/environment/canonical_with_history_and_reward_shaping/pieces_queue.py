from collections import deque
from typing import Optional
from domain.pieces import Piece


class PiecesQueue:
    def __init__(self, length: int) -> None:
        self.queue: deque[list[Piece]] = deque(maxlen=length)

    def push(self, pieces: list[Piece]) -> None:
        self.queue.append(pieces)

    def peek(self) -> list[Piece]:
        return self.queue[-1] if self.queue else None

    def peek_on_idx(self, idx: int) -> list[Piece]:
        try:
            return self.queue[idx]
        except IndexError:
            return None

    def __len__(self) -> int:
        return len(self.queue)
