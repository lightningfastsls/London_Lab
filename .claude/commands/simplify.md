# Simplify Code

Review and simplify recent code changes.

Look at the code that was just written or modified and:

1. **Remove Unnecessary Complexity**
   - Flatten nested conditionals where possible
   - Remove redundant checks or operations
   - Simplify boolean expressions

2. **Consolidate Duplicate Logic**
   - Extract repeated code into functions
   - Use list comprehensions where clearer
   - Combine similar operations

3. **Improve Naming**
   - Ensure variable names are descriptive
   - Function names should describe what they do
   - Class names should describe what they represent

4. **Type Hints**
   - Add missing type hints
   - Ensure return types are specified

5. **Docstrings**
   - Add docstrings to public functions
   - Keep them concise - one line if possible

## Rules

- Keep functionality identical - only improve code quality
- Keep diffs small and reviewable
- Run verification after simplification
- Preserve existing test coverage

## Output Format

```
SIMPLIFICATION REPORT

Files Modified:
- path/to/file.py

Changes Made:
1. [What was simplified and why]
2. [Another change]

Verification:
- [x] py_compile passed
- [x] Tests still pass
```

Focus area (optional): $ARGUMENTS
