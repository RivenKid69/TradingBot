# File Ownership and Permissions

This document outlines ownership and required permissions for major parts of the repository.

## File Ownership

- Source code (`*.py`, `*.pyx`, `*.cpp`, `*.h`) — owned by the **Core Development team**.
- Documentation under `docs/` — owned by the **Documentation team**.
- Continuous integration workflows in `.github/workflows/` — owned by the **DevOps team**.

Ownership is enforced via the repository's `CODEOWNERS` file.

## Required Permissions

| Role | Permissions |
| ---- | ----------- |
| Contributors | Read access, submit pull requests for review |
| Core Development team | Write access to source code files |
| Documentation team | Write access to `docs/` |
| DevOps team | Write access to `.github/workflows/` and CI settings |
| Repository administrators | Manage branch protection rules and secrets |

## CI/CD Editing Restrictions

- Branch protection should block direct pushes to `main` and require reviews from code owners.
- Changes to `.github/workflows/` must be approved by the DevOps team.
- Repository settings should limit who can modify CI/CD secrets and runners.
