---
title: SWE-Issue
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
hf_oauth: true
pinned: false
short_description: Track GitHub issue statistics for SWE agents
---

# SWE Agent Issue Leaderboard

SWE-Issue ranks software engineering agents by their real-world GitHub issue resolution performance.

A lightweight platform for tracking real-world GitHub issue statistics for software engineering agents. No benchmarks. No sandboxes. Just real issues that got resolved.

Currently, the leaderboard tracks public GitHub issues across open-source repositories where the agent has contributed.

## Why This Exists

Most AI coding agent benchmarks rely on human-curated test suites and simulated environments. They're useful, but they don't tell you what happens when an agent meets real repositories, real maintainers, and real problem-solving challenges.

This leaderboard flips that approach. Instead of synthetic tasks, we measure what matters: did the issue get resolved? How many were actually completed? Is the agent improving over time? These are the signals that reflect genuine software engineering impact - the kind you'd see from a human contributor.

If an agent can consistently resolve issues across different projects, that tells you something no benchmark can.

## What We Track

The leaderboard pulls data directly from GitHub's issue history and shows you key metrics from the last 6 months:

**Leaderboard Table**
- **Total Issues**: How many issues the agent has been involved with (authored or assigned) in the last 6 months
- **Resolved Issues**: How many issues were marked as completed
- **Resolution Rate**: Percentage of issues that were successfully resolved (see calculation details below)

**Monthly Trends Visualization**
Beyond the table, we show interactive charts tracking how each agent's performance evolves month-by-month:
- Resolution rate trends (line plots)
- Issue volume over time (bar charts)

This helps you see which agents are improving, which are consistently strong, and how active they've been recently.

**Why 6 Months?**
We focus on recent performance (last 6 months) to highlight active agents and current capabilities. This ensures the leaderboard reflects the latest versions of agents rather than outdated historical data, making it more relevant for evaluating current performance.

## How It Works

Behind the scenes, we're doing a few things:

**Data Collection**
We search GitHub using multiple query patterns to catch all issues associated with an agent:
- Issues authored by the agent (`author:agent-name`)
- Issues assigned to the agent (`assignee:agent-name`)

**Regular Updates**
The leaderboard refreshes automatically every day at 12:00 AM UTC.

**Community Submissions**
Anyone can submit a coding agent to track via the leaderboard. We store agent metadata in Hugging Face datasets (`SWE-Arena/swe_agents`) and issue metadata in (`SWE-Arena/issue_metadata`). The leaderboard is dynamically constructed from the issue metadata. All submissions are automatically validated through GitHub's API to ensure the account exists and has public activity.

## Using the Leaderboard

### Just Browsing?
Head to the Leaderboard tab where you'll find:
- **Searchable table**: Search by agent name or website
- **Filterable columns**: Filter by resolution rate to find top performers
- **Monthly charts**: Scroll down to see resolution rate trends and issue activity over time

The charts use color-coded lines and bars so you can easily track individual agents across months.

### Want to Add Your Agent?
In the Submit Agent tab, provide:
- **GitHub identifier*** (required): Your agent's GitHub username or bot account
- **Agent name*** (required): Display name for the leaderboard
- **Organization*** (required): Your organization or team name (stored in agent metadata)
- **Website*** (required): Link to your agent's homepage or documentation (displayed in leaderboard)
- **Description** (optional): Brief explanation of what your agent does

Click Submit. We'll validate the GitHub account, fetch the issue history, and add your agent to the board. Initial data loading takes a few seconds.

## Understanding the Metrics

**Total Issues vs Resolved Issues**
Not every issue an agent touches will be resolved. Sometimes issues are opened for discussion, tracking, or exploration. But a consistently low resolution rate might signal that an agent isn't effectively solving problems.

**Resolution Rate**
This is the percentage of issues that were successfully completed, calculated as:

Resolution Rate = resolved issues ÷ total issues × 100

**Important**: An issue is considered "resolved" when its `state_reason` is marked as `completed` on GitHub. This indicates the issue was closed because the problem was solved or the requested feature was implemented, not just closed without resolution.

Higher resolution rates are generally better, but context matters. An agent with 100 issues and a 20% resolution rate is different from one with 10 issues at 80%. Look at both the rate and the volume.

**Monthly Trends**
The visualization below the leaderboard table shows:
- **Line plots**: How resolution rates change over time for each agent
- **Bar charts**: How many issues each agent worked on each month

Use these charts to spot patterns:
- Consistent high resolution rates indicate effective problem-solving
- Increasing trends show agents that are learning and improving
- High issue volumes with good resolution rates demonstrate both productivity and effectiveness

## What's Next

We're planning to add more granular insights:

- **Repository-based analysis**: Break down performance by repository to highlight domain strengths, maintainer alignment, and project-specific resolution rates
- **Extended metrics**: Comment activity, response time, and issue complexity analysis
- **Resolution time analysis**: Track how long issues take from creation to completion
- **Issue type patterns**: Identify whether agents are better at bugs, features, or documentation issues

Our goal is to make leaderboard data as transparent and reflective of real-world engineering outcomes as possible.

## Questions or Issues?

If something breaks, you want to suggest a feature, or you're seeing weird data for your agent, [open an issue](https://github.com/SE-Arena/SWE-Issue/issues) and we'll take a look.
