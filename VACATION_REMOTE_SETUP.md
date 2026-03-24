# Vacation Remote Setup Guide

**Time needed:** ~30 minutes
**What you get:** Full access to your laptop from your phone, exactly as if you were sitting in front of it

---

## The Setup (do all of this before vacation)

### Step 1: Install Tailscale (5 min)

Tailscale creates a private network between your devices. Your laptop and phone can see each other from anywhere in the world, no port forwarding or static IP needed.

**On the laptop:**
1. Go to https://tailscale.com — sign up (free for personal use)
2. Download the Windows client, install, sign in
3. Note your laptop's Tailscale IP (looks like `100.x.y.z`) — you'll see it in the system tray icon

**On your phone:**
1. Install Tailscale from App Store / Google Play
2. Sign in with the same account
3. You should see your laptop listed

**Test it:** From your phone, open the Tailscale app and verify the laptop shows as "connected."

### Step 2: Install Chrome Remote Desktop (5 min)

This lets you see and control your laptop screen from your phone.

**On the laptop (Chrome browser):**
1. Go to https://remotedesktop.google.com/access
2. Click "Set up remote access"
3. Install the Chrome Remote Desktop extension + host app
4. Give your computer a name and set a PIN (6+ digits)
5. It should show as "Online"

**On your phone:**
1. Install "Chrome Remote Desktop" app from App Store / Google Play
2. Sign in with the same Google account
3. You should see your laptop listed

**Test it:** Connect from your phone right now. You should see your laptop screen. Tap to click, pinch to zoom. The keyboard button gives you a virtual keyboard.

### Step 3: Laptop Power Settings (2 min)

The laptop must stay awake while the lid is closed.

1. Open **Settings → System → Power & battery**
2. Or: **Control Panel → Hardware and Sound → Power Options → Choose what closing the lid does**
3. Set "When I close the lid" to **Do nothing** (for both "On battery" and "Plugged in")
4. Set "Turn off the screen" to **Never** (or a long time like 3 hours)
5. Set sleep to **Never** while plugged in
6. **Keep the laptop plugged in** during your vacation

### Step 4: Pull Repo on Laptop (5 min)

```powershell
cd C:\your\workspace
git clone https://github.com/your-username/mickey_london_lab.git
cd mickey_london_lab

# Set up the venv (if not already done)
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt  # or however you install deps

# Verify everything works
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m pytest usv_language/ -v
```

### Step 5: Verify Claude Code + arscontexta Work (5 min)

```powershell
cd mickey_london_lab
claude
# Check that session-orient hook fires
# Check that /stats works
# Check that notes/ are accessible
# Run /verify to confirm vault health
# Exit
```

### Step 6: Copy WAV Files (if needed)

Your ~6,500 WAV files probably aren't in git. If you need them on the laptop for detection work:

- **Easiest:** External USB drive, copy the `5970 USV/` folder
- **Or:** Network share / cloud sync
- **If you DON'T need WAVs for the vacation tasks** (null models + probing work on synthetic data): skip this entirely

### Step 7: Dry Run (5 min)

Do the full vacation workflow once while sitting at both devices:

1. **On the laptop:** Open Windows Terminal, start Claude Code
2. **Walk away, pick up your phone**
3. **On phone:** Open Chrome Remote Desktop app → connect to laptop
4. **On phone:** You should see the Windows Terminal with Claude Code running
5. **On phone:** Tap keyboard, type a message to Claude, verify it works
6. **On phone:** Pinch-zoom on the terminal area to make text readable

---

## The Daily Vacation Workflow

### Morning — Laptop (15 min)

Open 2-3 Windows Terminal tabs, one per task:

**Tab 1:**
```
claude
> I want to implement the null models module for information-theoretic analysis.
> Here's the spec: [paste from the plan I gave you]
> Enter plan mode first and show me your implementation plan.
```

**Tab 2:**
```
claude
> I want to implement the probing experiment framework.
> Here's the spec: [paste from the plan]
> Enter plan mode first.
```

Both sessions start planning. **Close the lid and go enjoy your vacation.**

### Mid-morning — Phone (5 min)

1. Open Chrome Remote Desktop on your phone
2. Connect to laptop
3. Tap on Tab 1 — see Claude's plan
4. If good: type `looks good, exit plan mode and start implementing`
5. Tap on Tab 2 — same thing
6. Disconnect

Claude is now coding. Your laptop is doing the work.

### Afternoon — Phone (10 min)

1. Reconnect via Chrome Remote Desktop
2. Check Tab 1 — see what Claude built
3. Type: `run the master-reviewer agent on everything you just implemented`
4. Check Tab 2 — same
5. Disconnect while reviewers run

### Late afternoon — Phone (10 min)

1. Reconnect
2. Read the master-reviewer feedback
3. Type: `fix the issues the reviewer found, then run pytest usv_language/ -v`
4. Disconnect

### Evening — Laptop (optional, 30 min if you feel like it)

```
# Check the implementations
# If you're happy, commit:
git add usv_language/analysis/
git commit -m "feat: null models + information theory framework"

# Feed findings into arscontexta:
/seed "Null models implementation complete. 5 generator types:
shuffled, Markov-k, renewal, HMM, phase-randomized.
Used Clauset 2009 MLE for Zipf. Miller-Madow for entropy."

/pipeline
```

### Days You Skip

**Nothing breaks.** The Claude Code sessions will eventually time out or hit the context limit. When you come back, just start new sessions. Your code is saved in files, your git history tracks everything.

---

## Tips for Phone Usage

### Chrome Remote Desktop on Phone

- **Pinch to zoom** on the terminal area — you'll need this, terminal text is small
- **Tap = click**, long press = right click
- **Keyboard icon** at the bottom gives you a virtual keyboard
- **Trackpad mode** (toggle in the hamburger menu) — lets you drag a cursor instead of direct tap. Better for precise clicking.
- **Landscape mode** is way more usable than portrait for terminal work

### Typing Efficiency

You'll be typing short commands, not essays. Typical phone inputs:

- `looks good, implement it`
- `run master-reviewer on this`
- `fix the issues and run tests`
- `show me the test results`
- `commit this to feature/null-models branch`

### If Claude Code Session Dies

If a session times out or crashes while you're away, no work is lost — files are already saved to disk. Just:

1. Connect from phone
2. Open a new terminal tab
3. Type `claude --continue` to resume, or start fresh with context

### If the Laptop Goes to Sleep Despite Settings

This is the #1 risk. To be safe:

- **Use a mouse jiggler app** (like "Move Mouse" from Microsoft Store — free). It simulates tiny mouse movements to prevent sleep.
- **Or:** Create a simple scheduled task that runs a trivial PowerShell command every 5 minutes:
  ```
  # In Task Scheduler, create a task that runs every 5 min:
  powershell -Command "[System.Windows.Forms.Cursor]::Position = [System.Windows.Forms.Cursor]::Position"
  ```

---

## What About Tailscale vs. Just Chrome Remote Desktop?

You might wonder: Chrome Remote Desktop works over the internet anyway, why do I need Tailscale?

**You might not.** Chrome Remote Desktop alone will work in most cases. Tailscale adds:

- **Backup access:** If Chrome RD glitches, you can SSH in via Tailscale as a fallback (install OpenSSH Server on Windows)
- **Faster connection:** Direct peer-to-peer instead of going through Google's relay
- **File transfers:** You can access the laptop's files directly via Tailscale's network

If you want to keep it minimal: **just install Chrome Remote Desktop and skip Tailscale.** Add Tailscale later only if you run into connection issues.

---

## Checklist

Before leaving:

- [ ] Chrome Remote Desktop installed on laptop + phone, tested
- [ ] (Optional) Tailscale installed on laptop + phone, tested
- [ ] Laptop power: lid close = do nothing, sleep = never (plugged in)
- [ ] Laptop will be plugged in during vacation
- [ ] Repo cloned and working on laptop
- [ ] Claude Code + arscontexta verified on laptop
- [ ] pytest passes on laptop
- [ ] WAV files copied (only if needed)
- [ ] Dry run from phone completed
- [ ] Task specs ready (null models + probing plans from our planning session)
- [ ] (Optional) Mouse jiggler installed
