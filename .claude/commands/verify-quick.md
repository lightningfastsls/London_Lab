# Quick Verification

Run a quick verification on all changed Python files.

## Steps
1. Find all Python files that have been modified (use `git status`)
2. Run `py_compile` on each modified .py file
3. If tests exist for the modified files, run them with pytest
4. Report results concisely

## Commands
```powershell
.\.venv\Scripts\python.exe -m py_compile <file>
.\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short
```

Report any errors clearly and suggest fixes.
