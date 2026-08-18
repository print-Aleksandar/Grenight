from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from domain.board_initialization import create_initial_board
from domain.requests import MoveRequest, ValidMovesPlayerRequest, ValidMovesPieceRequest
from domain.responses import ValidMovesPlayerResponse, ValidMovesPieceResponse
from domain.dtos import (MoveRequestDTO, MoveResponseDTO,
                         ValidMovesPlayerRequestDTO, ValidMovesPlayerResponseDTO,
                         ValidMovesPieceRequestDTO, ValidMovesPieceResponseDTO,
                         InitialBoardResponseDTO, get_piece_from_dto, get_dto_from_piece)
from domain.exceptions import ApiValidatorException
from application.game_service import (make_move, reject_move,
                                      gather_valid_moves_color, gather_valid_moves_piece)
from api.validators import (non_existent_valid_piece_with_uid_exception,
                            non_existent_board_position_exception,
                            player_not_on_turn_exception,
                            player_playing_with_enemy_pieces_exception,
                            player_not_on_turn_pieces_exception,
                            player_wants_moves_for_enemy_pieces_exception,
                            player_not_on_turn_piece_exception,
                            player_wants_moves_for_enemy_piece_exception)


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/get_initial_board")
def get_initial_board() -> InitialBoardResponseDTO:

    return InitialBoardResponseDTO(
        pieces=[get_dto_from_piece(piece) for piece in create_initial_board()]
    )


@app.post("/api/play_turn")
def play_turn(request_arg: MoveRequestDTO) -> MoveResponseDTO:

    request = MoveRequest(
        pieces=[get_piece_from_dto(piece_dto) for piece_dto in request_arg.pieces],
        uid=request_arg.uid,
        position=request_arg.position,
        is_white_on_turn=request_arg.is_white_on_turn,
        is_from_white_player=request_arg.is_from_white_player,
        is_current_move_promotion=request_arg.is_current_move_promotion,
        promote_to=request_arg.promote_to
    )

    validators = [
        lambda: non_existent_valid_piece_with_uid_exception(request),
        lambda: non_existent_board_position_exception(request),
        lambda: player_not_on_turn_exception(request),
        lambda: player_playing_with_enemy_pieces_exception(request)
    ]

    for validator in validators:
        try:
            validator()

        except ApiValidatorException as e:
            response = reject_move(request, e.__name__)
            return MoveResponseDTO(
                pieces=[get_dto_from_piece(piece) for piece in response.pieces],
                is_game_finished=response.is_game_finished,
                is_draw=response.is_draw,
                is_white_winner=response.is_white_winner,
                is_white_on_turn=response.is_white_on_turn,
                is_next_move_promotion=response.is_next_move_promotion,
                exception_message=response.exception_message,
                is_enemy_in_check=response.is_enemy_in_check
            )

    response = make_move(request)
    return MoveResponseDTO(
        pieces=[get_dto_from_piece(piece) for piece in response.pieces],
        is_game_finished=response.is_game_finished,
        is_draw=response.is_draw,
        is_white_winner=response.is_white_winner,
        is_white_on_turn=response.is_white_on_turn,
        is_next_move_promotion=response.is_next_move_promotion,
        exception_message=response.exception_message,
        is_enemy_in_check=response.is_enemy_in_check
    )

@app.post("/api/get_color_valid_moves")
def get_color_valid_moves(arg_request: ValidMovesPlayerRequestDTO) -> ValidMovesPlayerResponseDTO:

    request = ValidMovesPlayerRequest(
        pieces=[get_piece_from_dto(piece_dto) for piece_dto in arg_request.pieces],
        is_from_white_player=arg_request.is_from_white_player,
        is_white=arg_request.is_white,
        is_white_on_turn=arg_request.is_white_on_turn,
    )

    validators = [
        lambda: player_not_on_turn_pieces_exception(request),
        lambda: player_wants_moves_for_enemy_pieces_exception(request),
    ]

    for validator in validators:
        try:
            validator()

        except ApiValidatorException as e:
            response = ValidMovesPlayerResponse(None, e.__name__)
            return ValidMovesPlayerResponseDTO(
                uids_with_valid_moves=response.uids_with_valid_moves,
                exception_message=response.exception_message
            )

    response = gather_valid_moves_color(request)
    return ValidMovesPlayerResponseDTO(
        uids_with_valid_moves=response.uids_with_valid_moves,
        exception_message=response.exception_message
    )


@app.post("/api/get_piece_valid_moves")
def get_piece_valid_moves(arg_request: ValidMovesPieceRequestDTO) -> ValidMovesPieceResponseDTO:

    request = ValidMovesPieceRequest(
        pieces=[get_piece_from_dto(piece_dto) for piece_dto in arg_request.pieces],
        uid=arg_request.uid,
        is_from_white_player=arg_request.is_from_white_player,
        is_white_on_turn=arg_request.is_white_on_turn,
    )

    validators = [
        lambda: player_not_on_turn_piece_exception(request),
        lambda: player_wants_moves_for_enemy_piece_exception(request),
    ]

    for validator in validators:
        try:
            validator()

        except ApiValidatorException as e:
            response = ValidMovesPieceResponse(request.uid, None, e.__name__)
            return ValidMovesPieceResponseDTO(
                uid=response.uid,
                valid_moves=response.valid_moves,
                exception_message=response.exception_message
            )

    response = gather_valid_moves_piece(request)
    return ValidMovesPieceResponseDTO(
        uid=response.uid,
        valid_moves=response.valid_moves,
        exception_message=response.exception_message
    )
