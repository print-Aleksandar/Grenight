from domain.configs import ROWS, COLUMNS


class ActionEncoder:
    NUM_SQUARES = ROWS * COLUMNS

    NUM_PROMOTION_ACTIONS_ABSOLUTE = ((COLUMNS - 2) * 6 + 4 + 4) * 2

    NUM_PROMOTION_ACTIONS_CANONICAL = (COLUMNS - 2) * 6 + 4 + 4

    def __init__(self, is_canonical_version) -> None:

        self.is_canonical_version = is_canonical_version

        self.num_promotions_actions = (
            self.NUM_PROMOTION_ACTIONS_CANONICAL if self.is_canonical_version
            else self.NUM_PROMOTION_ACTIONS_ABSOLUTE
        )

        self.num_actions = (self.NUM_SQUARES * self.NUM_SQUARES) + self.num_promotions_actions

        self.black_promotion_offest = (
            0 if self.is_canonical_version
            else self.NUM_PROMOTION_ACTIONS_ABSOLUTE // 2
        )

    @staticmethod
    def position_to_square(position: tuple[int, int]) -> int:

        row, column = position
        return row * COLUMNS + column

    @staticmethod
    def square_to_position(square: int) -> tuple[int, int]:

        row = square // COLUMNS
        column = square % COLUMNS
        return row, column

    def encode(self, from_position: tuple[int, int],
               to_position: tuple[int, int]) -> int:

        from_square = self.position_to_square(from_position)
        to_square = self.position_to_square(to_position)

        return from_square * self.NUM_SQUARES + to_square

    def decode(self, action: int) -> tuple[tuple[int, int], tuple[int, int]]:

        from_square = action // self.NUM_SQUARES
        to_square = action % self.NUM_SQUARES

        from_position = self.square_to_position(from_square)
        to_position = self.square_to_position(to_square)

        return from_position, to_position

    def is_promotion_action(self, action: int) -> bool:
        return action >= self.NUM_SQUARES * self.NUM_SQUARES

    def encode_promotion(self, from_position: tuple[int, int],
                         to_position: tuple[int, int],
                         promote_to: int,
                         current_player_is_white: bool) -> int:

        from_col = from_position[1]
        dir_offset = to_position[1] - from_position[1]

        base_offset = 0 if current_player_is_white else self.black_promotion_offest

        if from_col == 0:
            dir_idx = 0 if dir_offset == 0 else 1
            offset = dir_idx * 2 + (promote_to - 1)

        elif from_col == COLUMNS - 1:
            offset = 4 + (COLUMNS - 2) * 6
            dir_idx = 0 if dir_offset == -1 else 1
            offset += dir_idx * 2 + (promote_to - 1)

        else:
            offset = 4 + (from_col - 1) * 6
            dir_idx = dir_offset + 1
            offset += dir_idx * 2 + (promote_to - 1)

        return (self.NUM_SQUARES * self.NUM_SQUARES) + base_offset + offset

    def decode_promotion(self, action: int,
                         current_player_is_white: bool) -> tuple[tuple[int, int], tuple[int, int], int]:

        offset = action - (self.NUM_SQUARES * self.NUM_SQUARES)

        base_offset = 0 if current_player_is_white else self.black_promotion_offest
        offset -= base_offset

        promote_to = (offset % 2) + 1
        offset //= 2

        if offset < 2:
            from_col = 0
            dir_offset = 0 if offset == 0 else 1

        elif offset >= 2 + (COLUMNS - 2) * 3:
            from_col = COLUMNS - 1
            last_offset = offset - (2 + (COLUMNS - 2) * 3)
            dir_offset = -1 if last_offset == 0 else 0

        else:
            mid_offset = offset - 2
            from_col = (mid_offset // 3) + 1
            dir_idx = mid_offset % 3
            dir_offset = dir_idx - 1

        if current_player_is_white or self.is_canonical_version:
            from_position = (ROWS - 2, from_col)
            to_position = (ROWS - 1, from_col + dir_offset)
        else:
            from_position = (1, from_col)
            to_position = (0, from_col + dir_offset)

        return from_position, to_position, promote_to
