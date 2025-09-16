#!/bin/bash

# Simple deployment script for GitHub Pages
# This script commits the built documentation and pushes to gh-pages branch

set -e

echo "🚀 Deploying documentation to GitHub Pages..."

# Ensure we're in the right directory
cd /Users/visheshyadav/Documents/GitHub/torchium

# Copy built docs to a temporary location
cp -r torchium/docs/build/html /tmp/torchium-docs

# Switch to gh-pages branch or create it
if git show-ref --verify --quiet refs/heads/gh-pages; then
    git checkout gh-pages
else
    git checkout --orphan gh-pages
    git rm -rf . 2>/dev/null || true
fi

# Copy documentation files
cp -r /tmp/torchium-docs/* .

# Add .nojekyll file to prevent Jekyll processing
echo "" > .nojekyll

# Commit and push
git add .
git commit -m "Update documentation - $(date)"
git push -f origin gh-pages

# Switch back to main branch
git checkout main

# Cleanup
rm -rf /tmp/torchium-docs

echo "✅ Documentation deployed successfully!"
echo "📄 View at: https://vishesh9131.github.io/torchium/"