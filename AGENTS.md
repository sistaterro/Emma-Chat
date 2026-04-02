# AGENTS.md

## Purpose

This file describes the recommended way to work in this repository for future agents, maintainers, and contributors. The goal is not to impose an idealized architecture, but to capture how the project is structured today and what the safest way is to evolve it.

When the request is to "update documentation", the expected scope in this project is:

- `README.md`
- `ui/Docs.html`
- `AGENTS.md`

## Project Summary

This repository implements a local chat application with RAG:

- FastAPI backend.
- Static frontend in `ui/*.html`.
- Main persistence in SQLite (`emma.db`).
- Source RAG files in `files/`.
- Chunks and embeddings in `chunks/`.
- Chat audit logs in `logs/chat_audit/`.
- Ollama integration for generation and auxiliary tasks.

Most of the business logic currently lives in `server.py`.

## Working Principles

- Understand the current flow before refactoring. This repo contains several pragmatic decisions and known technical debt; do not assume something is "wrong" just because it is not heavily modularized.
- Prefer small, safe, reversible changes. Avoid large refactors if the problem can be solved with a localized improvement.
- Keep sensitive logic in the backend. Permissions, validations, and access rules should not rely only on the frontend.
- The frontend should remain thin. The pages in `ui/` call the API with `fetch` and should not absorb complex business rules.
- Preserve the local/offline-first nature of the project. Do not introduce external infrastructure dependencies unless explicitly necessary.
- Prioritize real maintainability. If a rule, prompt, or flow is hard to find, centralize it.

## Structure And Responsibilities

- `server.py`
  - Main application entry point.
  - Contains auth, conversations, chat, files, indexing, retrieval, and auditing.
  - If new logic is added, keep it well scoped even if it still lives in this file.

- `prompts.py`
  - Canonical location for active system prompts.
  - Use pure functions such as `build_*_prompt(...)`.
  - Avoid embedding long prompts back into `server.py`.

- `ui/index.html`
  - Main home screen.
  - Should reflect visible permissions and available entry points by role.

- `ui/chat.html`
  - Main chat client.
  - Pay close attention to local state, rendering, and DOM cleanup when deleting or recreating conversations.

- `ui/upload.html`
  - RAG management screen.
  - The frontend may hide options by role, but the backend must remain the source of truth.

- `ui/admin.html`
  - Administrative UI.
  - Several parts were historically mocked; always verify what is connected to real backend behavior and what is not.

- `emma.db`
  - Local SQLite database.
  - Do not casually commit changes that only come from local usage or ephemeral data.

- `run.bat`
  - Windows startup script.
  - Should separate environment validation from execution as much as possible.

## Programming Methodology

### 1. Read First, Then Move Things

Before touching a feature:

- locate the backend endpoint involved;
- locate the HTML screen that consumes it;
- review whether there is persisted state in SQLite, JSON indexes, or files on disk;
- confirm whether any async processing is involved.

In this repo, many bugs come from interaction between frontend state, files, and asynchronous indexing, not just from one isolated function.

### 2. Backend First For Permissions

If user or role behavior changes:

- implement the restriction in the backend first;
- then hide or adapt the UI;
- never rely only on visual controls.

Current project roles:

- `admin`: can manage users and all RAGs.
- `user`: can use chat and manage their own `mine` RAGs.
- `read_only`: can only use chat and must not see or use upload.

### 3. Centralized Prompts

Prompts should live in `prompts.py`, not be distributed across multiple files.

Recommended convention:

- constants for shared rules;
- builder functions for dynamic prompts;
- clear names such as `build_rag_prompt`, `build_route_prompt`, `build_safety_prompt`.

Avoid prompt classes without real state.

### 4. Protect Visual State

The frontend is simple, so it needs extra care:

- if a view is hidden, clear the DOM if it may reappear with stale state;
- if a conversation or selection is deleted, reset local state explicitly;
- test scenarios with "only one item", because visual leftovers often appear there.

### 5. Defend Against Async Races

File indexing and other background tasks must assume that users can delete or modify resources while processing is still running.

Practical rule:

- before persisting derived results, verify that the original resource still exists;
- when maintaining auxiliary JSON indexes, prune orphaned entries when appropriate.

## Implementation Conventions

- Prefer pragmatic solutions over overengineering.
- If a change can be isolated in a helper function, do it.
- If a text or rule is hard to locate, move it to a canonical place.
- Keep names consistent with the current domain: `global`, `mine`, `owner_id`, `role`, `is_active`, and so on.
- Do not introduce empty abstractions such as managers or state-less classes if simple functions are enough.

## UX And Frontend

- Preserve the current visual language unless the goal is explicitly to redesign it.
- Solve responsiveness with measured, concrete changes, not complete rewrites.
- When cards or grids are conditionally shown by role, ensure stable centering and layout even when the number of visible items changes.
- If a screen does not apply to a role, hide it and block direct access when appropriate.

## Execution And Verification

Recommended workflow:

- use the local `.venv`;
- start with `run.bat` on Windows;
- validate quick syntax with:
  - `python -m py_compile server.py prompts.py`

Useful manual smoke tests after changes:

- login with `admin`, `user`, and `read_only`;
- correct card visibility in `index.html`;
- upload and delete of user-owned RAGs;
- `read_only` restrictions;
- user management from admin;
- chat creation, deletion, and recreation;
- index consistency when a file is deleted.

## Known Technical Debt

These debt items may exist consciously and should not be "fixed" without aligning scope first:

- `first use` flow and forced initial password change;
- `server.py` remains monolithic;
- parts of the admin UI may still need cleanup;
- `emma.db` often reflects local working state, not only schema.

## What To Do When Inheriting This Repo

Recommended order to understand it:

1. Read `server.py` to locate endpoints, auth, and the chat flow.
2. Read `prompts.py` to understand model behavior.
3. Review `ui/index.html`, `ui/chat.html`, `ui/upload.html`, and `ui/admin.html`.
4. Confirm the real schema in `emma.db`.
5. Review `files/`, `chunks/`, and `logs/chat_audit/` to understand auxiliary persistence.

## General Criterion

The best contribution in this project is usually to:

- make important things easier to find;
- harden backend behavior before polishing frontend behavior;
- reduce surprises;
- and leave each change easier to understand than before.
