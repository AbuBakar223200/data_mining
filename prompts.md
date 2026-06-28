# � Workflow Quick Reference

A guide to all available AI workflows in this repository.

---

## �🔧 Utility Workflows (Day-to-Day Coding)

| Workflow | What it Does | When to Use It |
|---|---|---|
| `/debug` | A scientific method for bugs: *Reproduce → Isolate → Hypothesize → Fix → Verify*. | When something breaks and you're stuck staring at the error. |
| `/optimize` | Safe refactoring. Makes code cleaner/faster *without breaking things*. | When code works but is slow, messy, or has duplication. |
| `/explain` | The "Teacher" - breaks down any code or concept so you understand the *why*. | When you find complex code (yours or library) and need to grok it. |
| `/unit_test` | Auto-generates `pytest` tests with edge cases (nulls, empty, boundaries). | After writing a function, to prove it works. |
| `/docstring` | Adds Google-style docstrings to your Python code. | Before committing code. Keeps it readable for "future you". |
| `/git_commit` | Writes clean `git commit` messages following Conventional Commits. | Every time you commit. (Your rule: "commit after small changes".) |

---

## 📊 Research & Publishing Workflows

| Workflow | What it Does | When to Use It |
|---|---|---|
| `/strategic_audit` | A "Principal Researcher" audit. Checks if your code *actually* matches your paper's claims. Finds gaps in reproducibility. | **Before** writing your paper. Run this on your `src/` and `results/` to see if you're ready. |
| `/reality_check` | An "Academic Advisor". Tells you what to cut and what to focus on, given your time/resource constraints. | When you're overwhelmed with ideas and need to prioritize. |
| `/write_findings` | Translates raw numbers into publishable "Results" and "Discussion" text. Emphasizes honesty about limitations. | After experiments are done, to write the "Findings" section. |
| `/reviewer` | A Statistics Reviewer. Checks if your metrics, splits, and claims are statistically sound. | Before final submission, to find holes a peer reviewer would. |
| `/auditor` | A QA Planner. Creates a full test plan for data pipelines and ML experiments. | To ensure your experiments are reproducible. |
| `/author` | A Research Writer. Helps draft sections of your paper. | When writing the paper itself. |
| `/paper_review` | A full Paper Reviewer simulation. | To get feedback on a draft before submission. |
| `/visualizer` | Creates publication-quality figures (confusion matrices, bar charts, etc.). | When you need to generate figures for the paper. |

---

## 🚀 How to Use Them (Productivity Tips)

### 1. Chain Them
```
/debug → /unit_test → /git_commit
```
Fix a bug, add a test for it, commit the fix.

### 2. Use `/strategic_audit` as Your "North Star"
- Run it *now* on your repository.
- It will tell you the "Delta" (gap) between your promises and your code.
- Use its output to create a TODO list.

### 3. Use `/reality_check` When Feeling Ambitious
- Before adding a new feature (e.g., "Let's try a Transformer!"), run `/reality_check`.
- It will tell you: *"You have 2 weeks. Focus on explaining RF results. Save Transformers for Future Work."*

### 4. End Every Session with `/git_commit`
- Just paste your `git diff` and ask `/git_commit` to write the message.
