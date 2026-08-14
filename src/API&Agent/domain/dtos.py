from pydantic import BaseModel
from domain.pieces import Piece
from domain.board_initialization import PIECES_CLASSES, PIECES_NUMBERS

class PieceDTO(BaseModel):
    uid: str
    is_white: bool
    position: tuple[int, int]
    had_first_move: bool
    class_number: int


def get_piece_from_dto(piece_dto: PieceDTO) -> Piece:
    return PIECES_CLASSES[piece_dto.class_number](uid=piece_dto.uid, is_white=piece_dto.is_white,
                                                  position=piece_dto.position, had_first_move=piece_dto.had_first_move,)


def get_dto_from_piece(piece: Piece) -> PieceDTO:
    return PieceDTO(uid=piece.uid, is_white=piece.is_white,
                    position=piece.position, had_first_move=piece.had_first_move,
                    class_number=PIECES_NUMBERS[type(piece)])


class MoveRequestDTO(BaseModel):
    pieces: list[PieceDTO]
    uid: str
    position: tuple[int, int]
    is_white_on_turn: bool
    is_from_white_player: bool
    is_current_move_promotion: bool
    promote_to: int | None = None


class ValidMovesPlayerRequestDTO(BaseModel):
    pieces: list[PieceDTO]
    is_from_white_player: bool
    is_white: bool
    is_white_on_turn: bool


class ValidMovesPieceRequestDTO(BaseModel):
    pieces: list[PieceDTO]
    uid: str
    is_from_white_player: bool
    is_white_on_turn: bool


class MoveResponseDTO(BaseModel):
    pieces: list[PieceDTO]
    is_game_finished: bool
    is_draw: bool
    is_white_winner: bool
    is_white_on_turn: bool
    is_next_move_promotion: bool
    exception_message: str | None = None


class ValidMovesPlayerResponseDTO(BaseModel):
    uids_with_valid_moves: list[tuple[int, int]] | None
    exception_message: str | None = None


class ValidMovesPieceResponseDTO(BaseModel):
    uid: str
    valid_moves: list[tuple[int, int]] | None
    exception_message: str | None = None
