import numpy as np
from domain.pieces import Piece
from domain.requests import MoveRequest, AgentMoveRequest
from domain.exceptions import GrenightException
from domain.board_initialization import create_initial_board
from application.game_service import make_move, gather_valid_moves_player
from application.board_getter import get_piece_by_position, get_piece_by_uid
from environment.action_encoder import ActionEncoder, PromotionActionEncoder
from environment.piece_plane_encoder import PiecePlaneEncoder


class GrenightEnvironment:

    def __init__(self, is_agent_white: bool):
        self.is_agent_white = is_agent_white

        self.state_encoder = PiecePlaneEncoder()
        self.action_encoder = ActionEncoder()
        self.promotion_action_encoder = PromotionActionEncoder()

        self.pieces: list[Piece] = []

        self.is_white_on_turn = True
        self.done = False

        self.is_promotion = False
        self.promotion_uid: str | None = None

    def reset(self) -> np.ndarray:
        self.pieces = create_initial_board()

        self.is_white_on_turn = True
        self.done = False

        self.is_promotion = False
        self.promotion_uid = None

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return self.state_encoder.encode(
            pieces=self.pieces,
            is_white_on_turn=self.is_white_on_turn,
            is_promotion=self.is_promotion,
        )

    def legal_actions(self) -> list[int]:

        if self.done:
            return []

        if self.is_promotion:
            return list(range(self.promotion_action_encoder.NUM_ACTIONS))

        request = AgentMoveRequest(
            pieces=self.pieces,
            is_for_white=self.is_white_on_turn,
            is_for_white_turn=self.is_white_on_turn,
            is_current_move_promotion=False,
        )

        uid_to_positions = gather_valid_moves_player(request)

        actions = []

        for uid, positions in uid_to_positions.items():

            piece = get_piece_by_uid(self.pieces, uid)

            for to_position in positions:

                action = self.action_encoder.encode(piece.position, to_position)

                actions.append(action)

        return actions

    def action_mask(self) -> np.ndarray:

        if self.is_promotion:
            num_actions = self.promotion_action_encoder.NUM_ACTIONS
        else:
            num_actions = self.action_encoder.NUM_ACTIONS

        mask = np.zeros(num_actions, dtype=bool)

        mask[self.legal_actions()] = True

        return mask

    def step(self, action: int):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first.")

        if self.is_promotion:
            return self._step_promotion(action)

        return self._step_move(action)

    def _step_move(self, action: int):

        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action}")

        from_position, to_position = self.action_encoder.decode(action)

        piece = get_piece_by_position(self.pieces, from_position)

        if piece is None:
            raise ValueError (f"Illegal action {action}: no piece at {from_position}")

        if piece.is_white != self.is_white_on_turn:
            raise ValueError(f"Illegal action {action}: piece belongs to the other player")

        request = MoveRequest(
            pieces=self.pieces,
            uid=piece.uid,
            position=to_position,
            is_white_on_turn=self.is_white_on_turn,
            is_from_white_player=self.is_white_on_turn,
            is_current_move_promotion=False,
            promote_to=None,
        )

        try:
            response = make_move(request)

        except GrenightException as e:
            raise ValueError(f"Move action {action} rejected by make_move: {type(e).__name__}")
        self.pieces = response.pieces

        if response.is_next_move_promotion:

            self.is_promotion = True
            self.promotion_uid = piece.uid

            next_state = self._get_state()

            info = {
                "is_enemy_in_check": response.is_enemy_in_check,
                "is_promotion": True,
            }

            return next_state, 0.0, False, info

        self.is_white_on_turn = response.is_white_on_turn
        self.done = response.is_game_finished

        reward = self._calculate_reward(
            response=response,
            is_agent_white=self.is_agent_white
        )

        next_state = self._get_state()

        info = {
            "is_enemy_in_check": response.is_enemy_in_check,
            "is_promotion": False,
        }

        return next_state, reward, self.done, info

    def _step_promotion(self, action: int):

        if action not in self.legal_actions():
            raise ValueError(f"Illegal promotion action {action}")

        if self.promotion_uid is None:
            raise RuntimeError("Environment is in promotion state but promotion_uid is None")

        promote_to = self.promotion_action_encoder.decode(action)

        promotion_piece = get_piece_by_uid(self.pieces, self.promotion_uid)

        if promotion_piece is None:
            raise RuntimeError(f"Promotion piece {self.promotion_uid} does not exist.")

        promotion_position = promotion_piece.position

        request = MoveRequest(
            pieces=self.pieces,
            uid=self.promotion_uid,
            position=promotion_position,
            is_white_on_turn=self.is_white_on_turn,
            is_from_white_player=self.is_white_on_turn,
            is_current_move_promotion=True,
            promote_to=promote_to,
        )

        try:
            response = make_move(request)

        except GrenightException as e:
            raise ValueError(f"Promotion action {action} rejected by make_move: {type(e).__name__}")

        self.pieces = response.pieces

        self.is_promotion = False
        self.promotion_uid = None

        self.is_white_on_turn = response.is_white_on_turn
        self.done = response.is_game_finished

        reward = self._calculate_reward(
            response=response,
            is_agent_white=self.is_white_on_turn
        )

        next_state = self._get_state()

        info = {
            "is_enemy_in_check": response.is_enemy_in_check,
            "is_promotion": False,
        }

        return next_state, reward, self.done, info

    @staticmethod
    def _calculate_reward(response, is_agent_white: bool) -> float:

        if not response.is_game_finished:
            return 0.0

        if response.is_draw:
            return 0.0

        if response.is_white_winner == is_agent_white:
            return 1.0

        return -1.0