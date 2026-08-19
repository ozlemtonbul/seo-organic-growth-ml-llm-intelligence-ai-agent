# Engineering Decision Log

## Why failed experiments are documented

A production ML project is not only the final model. It also includes the evidence used to reject models that should not be promoted.

This project explicitly retains the reasoning behind strategic forecast decisions.

## Examples

- A damped strategic ensemble was rejected after degrading important click metrics.
- A statistical champion approach was useful as a benchmark but removed from the final primary production route to preserve the ML-only product requirement.
- A 90-day lifecycle impression approach improved the baseline but still failed the absolute production threshold.
- A mixed statistical/ML CTR ensemble was replaced by a pure-ML ensemble.
- Direct strategic click targets are blocked when required tail scaling exceeds production guardrails.
- 365-day performance is not claimed as validated because historical coverage is insufficient.

This decision log is intended to make model promotion criteria inspectable and reproducible.
