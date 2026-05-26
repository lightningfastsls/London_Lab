# Deep research brief — Claude Code subagent authoring + evaluation in solo-dev orchestration

**Audience:** Claude.ai web (deep-research mode)
**Requester context:** Solo developer running a multi-agent orchestration loop on top of Claude Code (CLI). The orchestrator (a Claude Code main session) dispatches into a project-local catalog of ~15 specialized subagents via the `Agent` tool. We want to know whether community practice has converged on better authoring and evaluation conventions than our current homegrown approach — and where it has, what specifically we should copy.
**Deliverable wanted:** A ranked, citation-backed playbook of (a) authoring patterns for Claude Code subagents and Claude Agent SDK custom agents, and (b) evaluation patterns that validate dispatch behavior before relying on a new agent in production. Patterns we can apply, not a literature survey. Specifically — we want concrete examples of agent description / system-prompt structures that the publishers measured as effective, not theoretical advice on "good prompts."

---

## 1. Operating constraints (read these first; they filter the relevant literature)

- **Platform: Claude Code (CLI)**, not Claude.ai web or Claude API directly. The Agent tool dispatches subagents that get a *fresh context window* but inherit project CLAUDE.md + memory. Each subagent is one file under `.claude/agents/<name>.md` with YAML frontmatter (`name`, `description`, `tools`, `model`) plus a markdown body that becomes the system prompt.
- **The `description:` field is the dispatch routing signal.** When the orchestrator decides which agent to call for a task, it matches against descriptions. A description that's vague, overlapping with another agent's, or that doesn't say *when to use*, leads to mis-routing. Description quality is the single most leveraged property of an agent file.
- **Solo developer + main session, no human reviewer team.** Validation has to be cheap enough for one person to run before every agent ships. "Set up an eval harness with 200 graded tasks" is not actionable. "Run this 5-minute test before merging the agent" is.
- **Asymmetric model tiers.** Sonnet 4.6 (recently 4.7) for most subagents; Haiku 4.5 for read-only exploration; Opus 4.7 reserved for the orchestrator's own reasoning. We are not interested in patterns that assume every agent is Opus.
- **Closed-loop with an orchestrator.** Subagents return a single message; the orchestrator decides what to do next. We are not building a long-running autonomous loop. Patterns for autonomous agents (LangGraph, AutoGen, swarm patterns) are mostly out of scope unless they translate to single-shot subagent dispatch.
- **Hebrew/English mixed project.** Not directly relevant to authoring style but means we filter out "use emojis" / "use exclamation marks" stylistic advice — we want functional advice on what changes dispatch accuracy and output quality, not what makes the agent feel friendly.

---

## 2. What we already do (so you can critique it, not duplicate it)

### 2.1 Authoring conventions in our catalog

Every agent file in `.claude/agents/` follows this shape:

```markdown
---
name: <kebab-case-id>
description: <1–3 sentence dispatch routing hint that says what AND when>
tools: <comma-separated list — Read, Grep, Glob, Bash, Write, Edit, WebFetch, WebSearch>
model: sonnet | haiku
---

You are a <role>. <One-paragraph identity statement.>

## When invoked

1. <Step 1>
2. <Step 2>
3. ...

## What you must NOT do

- <Out-of-scope task 1>
- <Out-of-scope task 2>

## Context

<Standing priors / project-specific knowledge this agent always carries>
```

A current real example (`eval-runner.md`):

> **description:** Execute a pre-registered eval gate (N×M domain-runner invocations) and emit a FAIL/PASS verdict. Wraps the /eval-pregate skill with the operational priors needed for BFCL, curator, auditor, memory_recall, dev-agent ship-readiness runs — GPU swap timing, retry-on-flake, per-class threshold awareness, verdict-doc authorship in project voice. Use when an eval suite already exists and the work is "run it cleanly, write the verdict." Do not use to author new eval suites — that's prompt-iterator + test-architect territory.

The description deliberately includes a "Do not use for X" negation — naming the *adjacent* agent that handles the excluded case (here, prompt-iterator). This came out of one specific incident where two agents both matched the same task ("rerun curator with a tweaked prompt") and we had to sharpen the boundary.

We use `model: sonnet` as default. `haiku` only when the agent is genuinely read-only and small-context (e.g., a "find files matching pattern" agent). We have not yet used `model: opus` for any subagent.

### 2.2 Evaluation: the cold-read prediction test

Our internal validation method, documented in our memory as `feedback_cold_read_via_fresh_agent.md`, is:

> Dispatch a fresh-context subagent (the generic general-purpose agent) with ONLY the candidate agent's `description:` line — not the body. Ask them to: (a) predict scope, (b) write 3 example dispatch tasks, (c) list 3 tasks they would NOT dispatch for, (d) rate clarity 1–5, (e) flag any ambiguity. Compare their predictions to intent. If their "wouldn't dispatch for" list matches your intended exclusions without prompting, the description is sharp.

**Why it works:** the fresh agent simulates the orchestrator-at-dispatch-time picking from a catalog — which is the actual test the description has to pass. The author of the description cannot honestly cold-read their own work (they've absorbed the body), so the test has to be delegated.

**What it caught last time we ran it (5 new agents at once):** one real overlap risk (eval-runner ↔ prompt-iterator both matched "rerun curator with a tweaked prompt") and one fuzzy phrase ("measurable gates" — the fresh reader did not know what threshold was meant). Both were fixed before merge.

**Open questions about this method we want answered:** is it sufficient? Should it be paired with a behavioral test (actually dispatch the agent on 3 canned tasks and grade output)? Is there a known better methodology in the community?

### 2.3 Tool-permission convention

Each agent gets the minimum tool set for its job. Read-only agents (`data-archaeologist`, `code-reviewer`) have no `Write` or `Edit`. Subagents that author files (`handoff-writer`, `test-architect`) have `Write` + `Edit`. None get blanket `Bash` access without justification.

We do not use the `disallowedTools` field. We do not use the `permissionMode` field. We do not use the `skills` field to preload skills into the subagent — instead each subagent's body re-states the priors it needs, because we want subagents to be self-contained.

### 2.4 The mandatory-reviewer-pass policy gap

Our project's CLAUDE.md added a 2026-05-17 rule: "any net-new code file ≥50 LOC must get a reviewer-agent pass before being declared done." Today (2026-05-18) we added 5 new agent files (each ~80–150 lines of system prompt + frontmatter) and silently skipped the rule on the implicit argument that prompt config is not code.

The cold-read test we ran *is* a review pass — but it's a different instrument from `code-reviewer`. We need to know:

- Does the community treat agent / skill files as code (and run code-review on them) or as prompt config (and run prompt-eval on them)?
- Is there a published validation protocol specifically for *agent dispatch correctness* — as distinct from prompt quality, output quality, or runtime behavior?

---

## 3. Research questions

### Q1 — Authoring: description structure that maximizes correct dispatch

Anthropic's [Claude Code subagent docs](https://code.claude.com/docs/en/sub-agents.md) say descriptions should answer "what + when" and be specific. That's the floor. We want the **measured** patterns above that floor.

1. **Are there published A/B comparisons** (Anthropic Cookbook, customer engineering blog, community posts) showing which description structures empirically improve dispatch accuracy in a multi-agent catalog? Specifically — does adding an explicit "Do not use for X" negation actually reduce mis-dispatch, or is it cargo-cult?
2. **What's the optimal description length?** We're currently at 2–4 sentences. Anthropic example subagents (`code-reviewer`, `debugger`, `data-scientist`) are 1–2 sentences. Is there a length / clarity sweet spot people have measured?
3. **Is there a recommended discipline for *overlapping* agents?** We have several reviewer agents (`code-reviewer`, `master-reviewer`, `security-reviewer`, `test-hardener`). Each has a distinct intent but the orchestrator could plausibly call any of them on the same input. How do other multi-agent setups disambiguate, and what's the failure mode when they don't?

### Q2 — Authoring: system-prompt body structure

1. **"You are X" framing vs. role-by-action framing.** Our agents open with "You are a senior backend architect..." — does this work better than starting with the task ("Your job is to design a database schema...")? Any published comparisons?
2. **Numbered "When invoked" steps vs. prose workflow.** We use numbered steps consistently. Anthropic's example subagents do too. But is there a measured benefit, or is it just convention?
3. **"What you must NOT do" sections.** These are our convention for out-of-scope behavior. Are they used in published agents? Do they actually reduce out-of-scope output, or is it more effective to just state the in-scope task tightly?
4. **Standing-context priors in the body.** Our agents include a `## Context` section with project-specific knowledge (e.g., `data-archaeologist` carries 8 schema traps the project has hit before). Is there a better pattern? Should this go in CLAUDE.md instead? Should it be loaded as a `skills:` reference?

### Q3 — Evaluation: validating an agent before relying on it

This is the most leveraged question. Anthropic publishes no formal subagent eval framework that we can find. We want to know what's actually in practice.

1. **Cold-read prediction test critique.** Is our method (§2.2) something that exists in the literature under a different name? What does it miss? What's a published alternative?
2. **Behavioral / golden-task evals for subagents.** Is there a standard pattern: "5 canned input tasks → expected output shape → graded by judge"? For each subagent? For the dispatch decision specifically?
3. **Inter-agent dispatch confusion matrix.** If we have 15 agents, the right metric might be "% of tasks routed to the correct agent" measured across a labeled dispatch dataset. Has anyone built this for Claude Code?
4. **Regression detection when editing an agent description.** If we tweak `code-reviewer.md`'s description, the change can silently affect dispatch routing for unrelated tasks. Is there a "shouldn't-degrade" suite pattern for agent catalogs?
5. **Cost / token-budget evals.** Subagents inherit a fresh context window — we don't see how big it actually got. Has anyone measured per-agent context bloat as a function of system-prompt size + carried context?

### Q4 — When code-review-style review is and isn't the right instrument for agent files

1. **Community practice.** Do mature multi-agent Claude Code projects treat `.claude/agents/*.md` and `.claude/skills/*/SKILL.md` as code (subject to code-review by a reviewer agent) or as prompt config (subject to prompt-eval and cold-read)?
2. **Hybrid approach.** Is there a documented pattern for "this part of the file gets code-reviewed, this part gets prompt-evaluated"? (E.g., frontmatter validity + tool permissions = code review; description + body = prompt eval.)
3. **CI gating.** Is there a published CI hook for agent file changes that runs the cold-read test + a schema check automatically? We'd consider adding one if there's prior art.

### Q5 — Catalog hygiene at scale

1. **Catalog growth limits.** Anthropic's docs don't say how many subagents is "too many." With 15 we already see overlap. Is there published guidance on when a catalog should be refactored / consolidated?
2. **Versioning agent files.** When we update an agent's behavior, do we tag a version? Do we keep the old version? Is there a pattern for graceful deprecation in a catalog?
3. **Discovery surface for the orchestrator.** The orchestrator picks an agent based on what surfaces in its context (the list of agent descriptions). Is there guidance on how to *order* the list, or whether to group agents by domain in the descriptions themselves, to bias dispatch correctly?

---

## 4. What "good" looks like in the response

For each of the 5 question groups, we want:

1. **The most concrete pattern** (with a citation — Anthropic cookbook, customer-engineering blog, vendor docs, community post, OSS repo). Not "people generally do X" — a specific example from a specific source.
2. **Why it works** (1–2 sentences on the mechanism, not just the recipe).
3. **The closest analog if there is no direct match** (e.g., for Q3 the closest analog is probably prompt-eval frameworks — point us at the specific ones that translate to subagent dispatch).
4. **An honest "no published evidence" verdict where applicable.** We treat unverifiable advice on agent design as worse than no advice — speculating about what *might* work pollutes our judgment. If something is genuinely unstudied, say so plainly and we'll fall back to our cold-read methodology with the limitations acknowledged.

Output format: one section per question group (Q1–Q5), 200–500 words each. Citations as full URLs inline. Total response length we can absorb: ~3000–5000 words.

---

## 5. Out of scope

- Generic prompt engineering tutorials. We have those.
- Single-agent / autonomous-loop patterns (LangGraph, AutoGen, CrewAI, swarm). We are not building one.
- Frontier-model fine-tuning advice. Our subagents run on Anthropic's hosted models, not local weights.
- Specific Anthropic-confidential information. We only want public sources you can cite.
- Tool-use / function-calling design beyond what's in the agent definition file. Our subagents either succeed on Anthropic's defaults or are scoped to read-only — we are not designing tool schemas.

---

## 6. Project background (short, only if you need it for grounding)

We are building "Cloudy Claude" — an intelligence + website platform for MHS (an Israeli auto-parts importer). The Madaf ERP is the system of record; we are the system of intelligence. The orchestrator catalog is one piece of a larger setup that also includes local-GPU agents for inference and a knowledge vault for state. None of this should change your answer — we want general patterns. We mention it only so "what kind of project is this" doesn't sit as a background question.

---

## 7. How to deliver back

Save your response as `2026-05-18-claude-subagent-authoring-and-evaluation-response.md` and send it back to the requester. We will then translate the actionable items into:

- Edits to `.claude/agents/*.md` files (description sharpening, tool-permission tightening, body restructuring).
- A new validation skill or CI hook if the eval-design question yields concrete prior art.
- A CLAUDE.md edit codifying the reviewer-pass-or-cold-read policy.

Citations are load-bearing in this translation step — we will read the cited sources before applying any change, so please prioritize cite-able patterns over plausible-sounding synthesis.
