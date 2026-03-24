# Token Usage Optimization

Strategies for keeping Claude Code sessions efficient and cost-effective.

## Proactive Strategies

1. **Use specialized agents to learn first**
   - Before implementing signal processing, run `dsp-reviewer` on reference code
   - Learn patterns once, implement correctly first time
   - Avoids: implement -> debug -> fix cycles

2. **Use Task tool with model="haiku" for exploration**
   - Codebase searches, file discoveries, pattern finding
   - Reserve sonnet/opus for implementation and review

3. **Don't re-read files already in context**
   - Check conversation history before using Read tool
   - Exception: If file changed since last read

4. **Read targeted, not speculatively**
   - Use Grep to find specific patterns, then Read only matches

## When to Suggest New Session

Suggest starting fresh conversation when:
- Completing major feature or phase (clean break point)
- After extensive exploration (20+ file reads)
- Switching to unrelated work area
- Conversation >50 exchanges

**How to suggest:** "We've completed [X]. This is a good time to start a new session for [Y] to optimize token usage."
