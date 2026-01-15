# Commit, Push, and Create PR

Commit all staged changes, push to remote, and create a pull request.

## Steps
1. Run `git status` to see what will be committed
2. Run `git diff --cached` to review staged changes
3. Create a commit with a clear, descriptive message
4. Push to the current branch (create remote branch if needed)
5. Create a PR using `gh pr create`

## Commit Message Format
- First line: short summary (50 chars max)
- Blank line
- Body: explain what and why (not how)

## PR Format
- Title: same as commit summary
- Body: bullet points summarizing changes + test plan

## Commands
```powershell
git add <files>
git commit -m "message"
git push -u origin <branch>
gh pr create --title "Title" --body "Summary"
```
