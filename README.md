# <img src="src/frontend/logo/logo_transparent.png" alt="Grenight logo" width="115px"> — Green Knight Chess Platform

Grenight started as a way to exercise the integration and implementation of multiple technologies, while working through the fundamentals of structural, object-oriented, and functional programming, and system design. Along the way came the realization that processing a single chess move can be treated much like a stateless math script — and that idea became the foundation of Grenight, the Green Knight Chess Platform.

The goal of the final product is a production-ready chess platform where players can play matches in their browser, either against each other or against my own agent. Given the complexity of the project — from building the chess logic and exposing it as a stateless API, to building an agent, integrating it with the frontend, and making the whole thing production-ready as platfrom — project will pass through several phases:

## 4x5 Board — API & Frontend (Phase 1: Finished)

Phase 1 focused on building the core chess logic and exposing it through a stateless API, together with a lightweight frontend for testing and interaction.

The API validates each requested move and returns the resulting board state when the move is legal. Invalid moves are rejected and the turn remains unchanged until a valid move is submitted. The move validation and board-state transitions have been tested through a variety of game scenarios.

Pawn promotion is also implemented as a two-step process: once a pawn reaches the promotion rank, the game pauses until the promotion is completed. Checkmate and stalemate detection are implemented as well and have been tested through informal game scenarios.

The current frontend is intentionally lightweight and serves primarily as a testing and development interface rather than the final product. Clicking a piece displays its initial valid moves based only on its movement rules, pin and king-safety validation happens when the move is submitted.

## Refactoring (Phase 1.1: Finished)

This phase focused on improving accuracy and optimizing the parts of the system that will have a larger impact in later phases. The main optimization was replacing deep copies with structural sharing, reusing unchanged piece instances and creating a new instance only when a piece changes position.

Accuracy was also improved in the API. Invalid moves no longer return `200 OK` with an unnecessarily unchanged board. They now return `409 Conflict` with an appropriate error message.

The frontend was updated to display only valid positions when a piece is clicked instead of showing its initial movement positions. A visual effect was also added when a `NonExistentValidMoveException` occurs.

A PvP and PvE game mode selection was added. The PvE mode currently uses an agent with a random policy. The integration between the frontend and backend will remain unchanged in the next phase when the random policy is replaced with the RL agent.

## RL Environment & Agents - Researching and Implementing (Phase 2: Currently)