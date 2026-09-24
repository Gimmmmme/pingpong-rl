# Project brief

The project started from two requirements in the local conversation record:

1. Build a reinforcement-learning table-tennis experiment from scratch: two fixed-base bimanual robot assemblies, a table and net, physical paddles, training and evaluation without simulator-only observations at inference, and a reproducible demo entry point.
2. Replace the first paddle prototype with a realistic table-tennis racket, vary the ball trajectory and serve conditions, and support both hands on both robot sides so hand selection is part of the policy experiment.

This repository implements those requirements as an independent Isaac Lab/PhysX project. The upstream simulator repository is used only as an installation and public asset source; its task registry and application code are not imported here.
