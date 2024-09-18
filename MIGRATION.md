# Instructions for migrating from git.ligo.org to github.com

We hope that migrating development from [git.ligo.org](https://git.ligo.org/lscsoft/bilby) to [GitHub](https://github.com/bilby-dev/bilby) will be relatively painless.
This document provides instructions for common tasks.

## `main` vs `master`

This is not specifically a difference between Gitlab and GitHub, but we took this opportunity to make the [switch](https://www.theserverside.com/feature/Why-GitHub-renamed-its-master-branch-to-main).
New PRs will automatically be into `main` and new clones will automatically checkout `main`.
If you have existing clones (and muscle memory), make sure you always use `main` as the starting point for new branches.

## Migrating development to GitHub:

- [create a fork of github.com/bilby-dev/bilby if you don't already have one](https://github.com/bilby-dev/bilby/fork)
- if git.ligo.org/lscsoft/bilby is your current origin remove `git remote remove origin`
- make github the new origin `git remote add origin git@github.com:bilby-dev/bilby.git`
- add your fork on github as a remote `git remote add mine git@github.com:${ME}/bilby.git`
- checkout and update `main` `git checkout origin/main && git pull`
- checkout the new feature branch `git checkout -b FEATURE-BRANCH-NAME`
- push your feature branch to your fork `git push --set-upstream mine FEATURE-BRANCH-NAME`

## Migrating an MR to GitHub:

This largely follows the above instructions for migrating development.

- [create a fork of github.com/bilby-dev/bilby if you don't already have one](https://github.com/bilby-dev/bilby/fork)
- checkout your feature branch `git checkout FEATURE-BRANCH-NAME`
- add your fork on github as a remote `git remote add mine git@github.com:${ME}/bilby.git`
- push your feature branch to your fork `git push --set-upstream mine FEATURE-BRANCH-NAME`
- open a PR to [github.com/bilby-dev/bilby](https://github.com/bilby-dev/bilby/compare)
- copy the description and summarise relevant discussion
- close the MR on [git.ligo.org](https://git.ligo.org/lscsoft/bilby/-/merge_requests)

## Migrating an issue to GitHub:

This should not be necessary, we plan to automatically bulk migrate all issues (open and closed.)
