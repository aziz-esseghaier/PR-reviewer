"""Connects the reviewer to GitHub and Slack: the GitHub Action.

live_pr.py turns a live pull request into the same Instance the benchmark uses, pipeline.py
runs every stage on it, comments.py writes the proven findings as inline comments, github.py
posts them (as comments only: it never approves or merges), slack.py sends a short message
about them, and commands.py adds `pr-reviewer action ...`. PR text and model output are
untrusted input: comments and messages escape them, and our code builds every link.
"""
