"""H0.6 exclusive-XTX lifecycle orchestrator.

H0.6 has one job: create a trustworthy exclusive-XTX measurement
window and prove the machine returns to its original working state.

Pieces:
  snapshot      - capture pre-state (GPU, PIDs, VRAM, services, listeners,
                  production health)
  lock          - global flock so a second evaluator refuses
  ownership     - PID -> GPU UUID via rocm-smi /proc (replaces the
                  H0.5 heuristic)
  allowlist     - never pkill: only known, operator-allowlisted services
                  may be stopped to free the XTX; everything else
                  blocks the run
  restore       - trap that restores the exact pre-state on any exit
                  (SUCCESS, FAILURE, TIMEOUT, SIGTERM, SIGINT, exception)
  orchestrator  - glues the above
  cli           - entry point (subcommand on rdna.h06.cli)
"""