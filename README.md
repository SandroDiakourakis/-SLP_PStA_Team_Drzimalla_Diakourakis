# -SLP_PStA_Team_Drzimalla_Diakourakis


**Kurs:** SLP PStA

**Team:** Fabian Drzimalla (1114573), Sandro Diakourakis (1115059)

## Git Workflow & Branch Naming Convention

To ensure a clean and consistent development workflow, please follow these Git conventions for this project.

### Branch Types & Naming Scheme

| Type | Prefix | Example |
|------|---------|----------|
| New Feature | `feature/` | `feature/svm-cli` |
| Experiment | `experiment/` | `experiment/gridsearch-rbf-kernel` |
| Bug Fix | `fix/` | `fix/cli-argparse` |
| Refactoring | `refactor/` | `refactor/data-loader-structure` |
| Documentation | `docs/` | `docs/update-readme` |

### Rules
- Use **lowercase** and **hyphens (-)** to separate words.  
- Keep branch names **short but descriptive** (e.g., `experiment/svm-c-10-gamma-0.01`).  
- Each branch should serve **one clear purpose**.  
- Always create a **Pull Request (PR)** before merging into `main`.  
- Delete merged branches to keep the repo tidy.

### Example Workflow
```bash
# Create a new branch for an SVM CLI feature
git checkout -b feature/svm-cli

# Work and commit changes
git add .
git commit -m "Add initial CLI for SVM training"

# Push to GitHub
git push -u origin feature/svm-cli

# After review, open a PR to merge into main
```

---

## Experiment Tracking

For all experiments, please document results in a separate file called `branch-history.md`.

Each entry should include:
- **Branch name**
- **Goal / Experiment description**
- **Date**
- **Results / Observations**

Example entry:
```markdown
### experiment/svm-gridsearch-rbf
- **Date:** 2025-10-12
- **Goal:** Test SVM with RBF kernel and parameter tuning
- **Result:** Best accuracy 91.4% (C=10, gamma=0.01)
- **Notes:** Consider scaling features differently for next run.
```

## Use virutal environments
To manage dependencies, please use a virtual environment. You can create one using `venv` or `conda`.
Example using `venv`:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```