# Task: Gather USV Pipeline & Skills Context for Prompt Optimization

## Goal
We're extending a STAR reasoning framework optimization (from arXiv:2602.21814) beyond the Cloudy Claude API prompts to cover the USV detection/classification pipeline and Claude Code skills. We need a map of every place where Claude (or another model) makes decisions that involve implicit constraint reasoning.

## What to gather

### 1. Skills Inventory
List every skill in `.claude/skills/` with:
- Skill name
- One-line description of what it does
- Whether it involves any prompt templates or reasoning instructions
- Full text of any system prompts or instruction blocks within the skill

### 2. USV Pipeline — AI Decision Points
Search the entire repo for places where the USV pipeline uses AI/ML inference or where Claude is prompted about DSP/signal processing decisions. This includes:
- Any prompt templates related to USV detection, classification, or analysis
- CNN model configuration and any comments/docs about implicit assumptions (e.g., frequency ranges, STFT parameters, threshold choices)
- Any scripts or notebooks where Claude is asked to reason about spectrogram analysis
- LMT (micecraft) integration points where behavioral classifications are made
- The DeepSqueak/BootSnap integration plans and any prompt templates for syllable classification

### 3. Agent Prompt Full Text
Output the **complete text** of every file in `.claude/agents/`:
- dsp-reviewer.md
- master-reviewer.md  
- pr-reviewer.md
- detection-validator.md
- streamlit-expert.md
- test-writer.md

### 4. Command Templates
Output the **complete text** of every file in `.claude/commands/`:
- implement.md
- verify.md
- commit-push-pr.md
- Any others present

### 5. Knowledge Pipeline Skills
Specifically output the full text of these skills (if they exist):
- reduce
- reflect
- reweave
- verify
- pipeline
- Any other skills that involve multi-step reasoning

### 6. ops/ Directory
Output the current contents of:
- ops/goals.md
- ops/reminders.md
- Any other files in ops/

## Output format
Create `usv-and-skills-audit.md` with all findings. Paste full file contents — don't summarize. 

## Important
Do NOT change any files. Read-only audit.
