class BoardConfig:
    def __init__(self, columns: int,
                 rows: int,
                 pawn_double_step: bool,
                 castling: bool) -> None:
        self.columns = columns
        self.rows = rows
        self.pawn_double_step = pawn_double_step
        self.castling = castling


BOARD_4x5_CONFIG = (4, 5, False, False)
BOARD_6x6_CONFIG = (6, 6, False, True)
BOARD_8x8_CONFIG = (8, 8, True, True)


# SELECTING BOARD CONFIGURATION
CURRENT_BOARD_CONFIG = BoardConfig(*BOARD_4x5_CONFIG)


COLUMNS = CURRENT_BOARD_CONFIG.columns
ROWS = CURRENT_BOARD_CONFIG.rows
PAWN_DOUBLE_STEP = CURRENT_BOARD_CONFIG.pawn_double_step
CASTLING = CURRENT_BOARD_CONFIG.castling
POSITIONS = []
[POSITIONS.extend([(y,x) for x in range(COLUMNS)]) for y in range(ROWS)]
