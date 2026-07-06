#!/usr/bin/env bash
#
# sync.sh — commit, sync between main/answer, and push both branches.
#
# Usage:
#   ./sync.sh "your commit message"
#
# What it does (run it from anywhere inside the repo):
#   1. Commits all current changes on the branch you're on.
#   2. Pushes that branch.
#   3. Merges your branch into the OTHER branch (main <-> answer) and pushes it.
#      Practice .py files stay branch-divergent thanks to merge=ours.
#   4. Returns you to the branch you started on.

set -euo pipefail

MSG="${1:-}"
if [ -z "$MSG" ]; then
  echo "Usage: $0 \"commit message\"" >&2
  exit 1
fi

# Make sure the merge=ours driver is registered (idempotent, needed for .gitattributes).
git config merge.ours.driver true

# Branch we're on, and the one to sync to.
CUR=$(git branch --show-current)
case "$CUR" in
  main)   OTHER="answer" ;;
  answer) OTHER="main" ;;
  *)
    echo "Expected to be on 'main' or 'answer', but you're on '$CUR'." >&2
    exit 1
    ;;
esac

# 1. Commit changes on the current branch (skip if there's nothing to commit).
git add -A
if git diff --cached --quiet; then
  echo "No changes to commit on $CUR."
else
  git commit -m "$MSG"
fi

# 2. Push the current branch.
git push origin "$CUR"

# 3. Sync into the other branch and push it.
git switch "$OTHER"
git merge --no-ff "$CUR" -m "Merge $CUR into $OTHER: $MSG"
git push origin "$OTHER"

# 4. Back to where you started.
git switch "$CUR"

echo "Done: committed on $CUR, merged into $OTHER, pushed both."
