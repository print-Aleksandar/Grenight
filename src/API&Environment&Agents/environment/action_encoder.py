from domain.board_configs import ROWS, COLUMNS


class ActionEncoder:

    NUM_SQUARES = ROWS * COLUMNS
    NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES

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

    def decode(self, action: int):

        from_square = action // self.NUM_SQUARES
        to_square = action % self.NUM_SQUARES

        return (
            self.square_to_position(from_square),
            self.square_to_position(to_square)
        )


class PromotionActionEncoder:

    NUM_ACTIONS = 4
    PROMOTION_CLASS_NUMBERS = [1, 2, 3, 4]

    def encode(self, promotion_index: int) -> int:
        if not 0 <= promotion_index < self.NUM_ACTIONS:
            raise ValueError("Invalid promotion action")

        return promotion_index

    def decode(self, action: int) -> int:
        if not 0 <= action < self.NUM_ACTIONS:
            raise ValueError("Invalid promotion action")

        return self.PROMOTION_CLASS_NUMBERS[action]
