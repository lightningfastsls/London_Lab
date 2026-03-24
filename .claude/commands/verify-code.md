# Verify Implementation

Run appropriate verification steps for the current implementation.

## Steps

1. **Syntax Check**: Run py_compile on all modified Python files
2. **Tests**: Run pytest on relevant test files
3. **Linting**: Run flake8 if configured
4. **Output Verification**: If applicable, verify outputs look correct

## Commands

```powershell
# Syntax check
.\.venv\Scripts\python.exe -m py_compile <file>

# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -v --tb=short

# Run specific tests
.\.venv\Scripts\python.exe -m pytest tests/test_<module>.py -v
```

## Output Format

```
VERIFICATION RESULTS

Syntax Check:
- [x] file1.py - OK
- [ ] file2.py - ERROR: [description]

Tests:
- [x] test_module.py - 5 passed
- [ ] test_other.py - 2 failed

Issues Found:
1. [Issue description]
   - Fix: [suggested fix]

Overall: PASS / FAIL
```

Focus area (optional): $ARGUMENTS
