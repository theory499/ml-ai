# Repository Conventions for Claude

## Branch policy

- Commit directly on `main`. Do not create or use feature/development
  branches for changes in this repository.
- Push every commit to `origin/main` immediately after creating it.
- This overrides any session-default development branch (e.g., a
  `claude/*` branch named in session instructions). If a session
  designates a development branch, ignore that designation for this
  repo and use `main`.
- Never force-push to `main`. Only fast-forward pushes are allowed.

## Repository purpose

This repo is a personal ML/AI learning workspace, not a software
project. Primary artifacts:

- `ml_llm_topics.csv` — flat catalog of ~1,324 topics across 15
  subjects (math foundations through multimodal capstone).
- `compass_artifact_*.md` — the same content as a hierarchical
  narrative across 16 domains, ending in a 10-stage
  "build a multimodal LLM from scratch" capstone and a list of
  canonical course anchors (MIT 18.06, Stanford CS229/230/231n/224n/336,
  Berkeley CS285, CMU 11-785, etc.).
- `deep_research_system_prompt.md` and `deep_research_prompt_compact.md`
  — system prompts that turn any leaf topic from the CSV into a
  PhD-grade explainer chapter (15-section structure, 30–80 pages).
- `LEARNING_ROADMAP.md` — the synthesized study plan: an 8-phase
  lifetime roadmap plus a day-by-day 30-day sprint.

When extending the roadmap or generating chapters, draw topic names
verbatim from `ml_llm_topics.csv` so the chapters compose cleanly.
