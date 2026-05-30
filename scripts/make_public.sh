#!/usr/bin/env bash
# Flip this repo from private to public.
# Run on or after 2026-06-13 (the day after BlueDot's "no public sharing
# before 12th June" deadline).
#
# Requires: gh CLI authenticated with admin access to this repo.
# Run with:   bash scripts/make_public.sh

set -e
REPO="viraajminhas/bluedot-tais-puzzle-1-solution"
TODAY=$(date +%Y-%m-%d)
DEADLINE="2026-06-12"

if [[ "$TODAY" < "2026-06-13" ]]; then
    echo "Today is $TODAY. BlueDot's deadline is $DEADLINE."
    echo "Wait until 2026-06-13 to publish."
    exit 1
fi

gh repo edit "$REPO" --visibility public --accept-visibility-change-consequences
gh repo edit "$REPO" --add-topic interpretability \
                     --add-topic ai-safety \
                     --add-topic mechanistic-interpretability \
                     --add-topic bluedot-impact
echo "Repo is now public. Pin it to your GitHub profile via the website."
