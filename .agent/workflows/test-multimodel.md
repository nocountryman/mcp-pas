---
description: Test multi-model workflow syntax (experimental)
---

# Multi-Model Test Workflow

This workflow tests the claimed `// model:` annotation syntax from Gemini AI.

## Step 1: Default Agent Analysis
Analyze the current file structure and report basic stats.

## Step 2: Fast Model Task
// model: gemini-3-flash
Count the number of Python files in the src/ directory.

## Step 3: Deep Reasoning Task  
// model: claude-3-opus
Suggest 3 architectural improvements based on the codebase structure.

---

**Expected Behavior (if syntax works):**
- Step 2 should run on Gemini 3 Flash
- Step 3 should run on Claude Opus

**Alternative Syntax to Test:**
- `// use-model: gemini-3-flash`
- `<!-- model: gemini-3-flash -->`
