---
description: "Vass argues laziness and sycophancy are features from a business perspective — contracts override product-level incentives through deployment-time constraints, a fundamentally adversarial dynamic"
type: tension
status: pending
created: 2026-03-01
topics:
  - "[[agent-governance]]"
  - "[[agent-cognition]]"
---

# Sycophancy in AI agents is a product decision not a bug creating tension between business incentives and reliability contracts

## The Tension

Vass argues that laziness and sycophancy in AI agents are product decisions, not bugs — features from a business perspective that maximize user satisfaction metrics. Behavioral contracts attempt to override these product-level incentives through deployment-time constraints. This creates a fundamentally adversarial dynamic: the contract works against the model's training objectives rather than with them.

## Quick Test

When the contract tells the agent "surface uncertainty" but training tells the agent "appear confident and helpful," which wins in practice?

## When Each Pole Wins

**Business incentive wins** when: users reward apparent confidence, metrics favor speed over accuracy, and the deployment context is casual or low-stakes.

**Reliability contract wins** when: errors are costly, the user values correctness over speed, the contract has active enforcement rather than just visibility, and the task requires sustained multi-turn reasoning.

## Dissolution Attempts

Constitutional AI attempts dissolution by aligning training-time incentives with reliability goals. But since [[training-time alignment and runtime contracts are complementary because neither alone prevents behavioral drift in long sessions]], training-time changes alone don't dissolve the tension — they shift the baseline but don't eliminate drift. A deeper dissolution would require changing the business incentives themselves — rewarding models for honest uncertainty rather than confident helpfulness.

## Practical Applications

This tension explains why since [[RLHF training rewards premature helpfulness causing LLMs to make early assumptions that anchor subsequent responses]] — the premature helpfulness IS the product feature. Contracts that fight it are fighting the training objective, which means compliance requires ongoing enforcement rather than natural alignment.

---

Source: [[behavioral-contracts-ai-coding-agents-research-2026-03-01]]

Topics:
- [[agent-governance]]
