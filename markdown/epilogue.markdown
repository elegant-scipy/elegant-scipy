# Epilogue

## Where to next?
- Parts of the SciPy Ecosystem we haven't covered
- Other books/websites/resources
- Conferences (and their recorded sessions - links)

## Contributing to the SciPy ecosystem
Why contribute?
Promises from preface:
We will help you get started in GitHub land
Later in Elegant SciPy we will show you how to contribute your new skills to the GitHub-hosted projects that comprise most of the scientific Python ecosystem.
- how to use git and github (a short, quick-reference style, guide - see below for topics)
- best practices (for...?)
- how to contribute to SciPy and related packages specifically (community rules?)

## Open Science
Why do open science?
- Reminder about open source licenses (refer to introduction)
- Open data
- Open access and preprints


## A quick-start guide to Git and GitHub

This is not intended to be a complete guide, there are plenty of great books out there on this topic.
If you are completely new to Git, we suggest you look at a beginner tutorial first (link - software carpentry http://software-carpentry.org/lessons/).
They will show you how to install Git, create a GitHub account, and understand the essential commands.
If you have some familiarity with Git then you can probably jump in here.
There are different ways of using Git and GitHub, for all sorts of reasons.
Here we're talking about how to contribute to an open source project using the most common approach (in our experience, anyway).

What are Git and GitHub?

Issues on GitHub
- A great place to start interacting with an open source project.

Fork the project on GitHub

Clone
- `$ git clone url`
- `cd` into directory

Branch
- `$ git checkout -b branchname`

Edit-add-commit cycle
- edit and save the file
- `$ git add filename(s)`
- `$ git commit -m "A note about what I changed"`

Push
`$ git push origin branchname`

Submit a pull request on GitHub
- Be polite
- Don't be offended by feedback
- Keep trying

Pulling and pushing (to your fork)
- pull vs fetch (and when they can be useful)

Dealing with remotes: upstream and origin
- Add a new remote
`$ git remote add upstream url`
- Check your remotes
`$ git remote -v`

Rebasing - keeping up to date with master
- `$ git checkout master`
- `$ git pull upstream master`
- `$ git checkout branchname`

Squash all commits into a single commit (not required, but makes rebasing quicker and easier)
- `$ git merge-base branchname master commit-hash`
- On all but the first line replace “pick” with “squash”

Start the actual rebase
`$ git rebase --interactive`
- If you get an error, you can edit the offending file then:
`$ git add file`
`$ git rebase --continue`

Force-push to update your pull request
`$ git push -f origin branchname`
Your pull request on GitHub should update automatically.

Tidy up:
`$ git branch -d branchname`
