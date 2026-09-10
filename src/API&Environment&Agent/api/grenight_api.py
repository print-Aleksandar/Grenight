import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from agent.grenight_agent import GrenightAgent
from domain.board_initialization import create_initial_board
from domain.configs import COLUMNS, ROWS, MAX_STEPS_PER_EPISODE
from domain.requests import MoveRequest, ValidMovesPieceRequest, AgentMoveRequest
from domain.dtos import (MoveRequestDTO, MoveResponseDTO,
                         ValidMovesPieceRequestDTO, ValidMovesPieceResponseDTO,
                         InitialBoardResponseDTO, get_piece_from_dto, get_dto_from_piece, AgentMoveRequestDTO)
from domain.exceptions import GrenightException
from application.game_service import (make_move,
                                      gather_valid_moves_piece)
from api.api_validators import (non_existent_valid_piece_with_uid_exception,
                                non_existent_board_position_exception,
                                player_not_on_turn_move_exception,
                                player_wants_to_play_with_enemy_pieces_exception,
                                player_not_on_piece_valid_moves_turn_exception,
                                player_wants_to_gather_valid_moves_for_enemy_piece_exception,
                                agent_not_on_turn_exception)
from domain.responses import MoveResponse
from environment.grenight_environment import GrenightEnvironment

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
    env.reset()

    return InitialBoardResponseDTO(
        pieces=[get_dto_from_piece(piece) for piece in create_initial_board()],
        is_white_on_turn=True
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
        lambda: player_not_on_turn_move_exception(request),
        lambda: player_wants_to_play_with_enemy_pieces_exception(request)
    ]

    for validator in validators:
        try:
            validator()

        except GrenightException as e:
            raise HTTPException(status_code=409, detail=type(e).__name__)

    try:
        response = make_move(request)

    except GrenightException as e:
        raise HTTPException(status_code=409, detail=type(e).__name__)

    return MoveResponseDTO(
        pieces=[get_dto_from_piece(piece) for piece in response.pieces],
        is_game_finished=response.is_game_finished,
        is_draw=response.is_draw,
        is_white_winner=response.is_white_winner,
        is_white_on_turn=response.is_white_on_turn,
        is_next_move_promotion=response.is_next_move_promotion,
        is_enemy_in_check=response.is_enemy_in_check
    )


@app.post("/api/get_piece_valid_moves")
def get_piece_valid_moves(request_arg: ValidMovesPieceRequestDTO) -> ValidMovesPieceResponseDTO:

    request = ValidMovesPieceRequest(
        pieces=[get_piece_from_dto(piece_dto) for piece_dto in request_arg.pieces],
        uid=request_arg.uid,
        is_from_white_player=request_arg.is_from_white_player,
        is_white_on_turn=request_arg.is_white_on_turn,
    )

    validators = [
        lambda: player_not_on_piece_valid_moves_turn_exception(request),
        lambda: player_wants_to_gather_valid_moves_for_enemy_piece_exception(request),
    ]

    for validator in validators:
        try:
            validator()

        except GrenightException as e:
            raise HTTPException(status_code=409, detail=type(e).__name__)

    response = gather_valid_moves_piece(request)
    return ValidMovesPieceResponseDTO(
        uid=response.uid,
        valid_moves=response.valid_moves
    )


@app.post("/api/agent_move")
def agent_move(request_arg: AgentMoveRequestDTO) -> MoveResponseDTO:

    request = AgentMoveRequest(
        pieces=[get_piece_from_dto(piece_dto) for piece_dto in request_arg.pieces],
        is_for_white=request_arg.is_for_white,
        is_for_white_turn=request_arg.is_for_white,
        is_current_move_promotion=request_arg.is_current_move_promotion
    )

    validators = [
        lambda: agent_not_on_turn_exception(request)
    ]

    for validator in validators:
        try:
            validator()

        except GrenightException as e:
            raise HTTPException(status_code=409, detail=type(e).__name__)

    try:
        response = agent_taking_action(request)

    except GrenightException as e:
        raise HTTPException(status_code=409, detail=type(e).__name__)

    return MoveResponseDTO(
        pieces=[get_dto_from_piece(piece) for piece in response.pieces],
        is_game_finished=response.is_game_finished,
        is_draw=response.is_draw,
        is_white_winner=response.is_white_winner,
        is_white_on_turn=response.is_white_on_turn,
        is_next_move_promotion=response.is_next_move_promotion,
        is_enemy_in_check=response.is_enemy_in_check
    )


env = GrenightEnvironment(is_canonical_version=False,
                          will_store_history_in_state=False,
                          will_do_reward_shaping=False)

agent = GrenightAgent(
    is_self_play=False,
    is_double_net=True,
    is_dueling_net=False,
    is_residual_net=True,
    is_bulk_update=False,
    rows=ROWS,
    columns=COLUMNS,
    num_actions=env.action_encoder.num_actions,
    num_planes=env.state_encoder.num_planes,
    device="cpu"
)

current_dir = Path(__file__).resolve().parent
checkpoint_path = current_dir / "../agent/implementations/ver50/p_101000/current_implementation_ep8000.pt"
checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu",weights_only=False)
agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
if agent.is_double_net:
    agent.target_net.load_state_dict(checkpoint["target_state_dict"])


def agent_taking_action(request: AgentMoveRequest) -> MoveResponse:
    d1 = {p.uid: p.position for p in request.pieces}
    d2 = {p.uid: p.position for p in env.pieces}

    if d1 != d2:
        env.load_pieces_absolute(request.pieces)

    action = agent.select_action(env.get_state(), env.action_mask(), 0.5)
    _, _, done, is_draw, _ = env.step(action)

    return MoveResponse(
        pieces=env.pieces,
        is_game_finished=done,
        is_draw=is_draw,
        is_white_winner=True if done and not is_draw else False,
        is_white_on_turn=env.is_white_on_turn,
        is_next_move_promotion=False,
        is_enemy_in_check=env.is_enemy_in_check
    )
