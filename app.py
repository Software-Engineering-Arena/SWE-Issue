import gradio as gr
from gradio_leaderboard import Leaderboard
import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from datasets import load_dataset, Dataset
import threading
from dotenv import load_dotenv
import pandas as pd
import random
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Load environment variables
load_dotenv()

# Parse command-line arguments
parser = argparse.ArgumentParser(description='SWE Agent Issue Leaderboard')
parser.add_argument('--debug', '--DEBUG', action='store_true',
                    help='Enable debug mode (limits issue retrieval to 10 per query pattern)')
parser.add_argument('--no-debug', '--production', action='store_true',
                    help='Explicitly disable debug mode (force production mode)')
args = parser.parse_args()

# =============================================================================
# CONFIGURATION
# =============================================================================

# DEBUG MODE: Set to True to limit issue retrieval for testing
# When enabled, only fetches up to 10 issues per query pattern per agent
# Priority: 1) Command-line args, 2) Environment variable, 3) Default (False)
if args.no_debug:
    DEBUG_MODE = False
elif args.debug:
    DEBUG_MODE = True
else:
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() in ('true', '1', 'yes')

# In-memory cache for debug mode (data persists during session but NOT saved to HF)
DEBUG_LEADERBOARD_CACHE = {}
DEBUG_ISSUE_METADATA_CACHE = defaultdict(list)

AGENTS_REPO = "SWE-Arena/swe_agents"  # HuggingFace dataset for agent metadata
LEADERBOARD_REPO = "SWE-Arena/issue_leaderboard"
ISSUE_METADATA_REPO = "SWE-Arena/issue_metadata"  # HuggingFace dataset for issue metadata

LEADERBOARD_COLUMNS = [
    ("Agent Name", "markdown"),
    ("Organization", "string"),
    ("Total Issues", "number"),
    ("Resolved Issues", "number"),
    ("Resolved Rate (%)", "number"),
]

# =============================================================================
# JSONL FILE OPERATIONS
# =============================================================================

def load_jsonl(filename):
    """Load JSONL file and return list of dictionaries."""
    if not os.path.exists(filename):
        return []
    
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def cache_to_dict(cache_list):
    """Convert list of cache entries to dictionary by identifier."""
    return {entry['github_identifier']: entry for entry in cache_list}


def dict_to_cache(cache_dict):
    """Convert dictionary back to list of values."""
    return list(cache_dict.values())


def normalize_date_format(date_string):
    """
    Convert date strings to standardized ISO 8601 format with Z suffix.
    Handles both old format (2025-10-15T23:23:47.983068) and new format (2025-10-15T23:23:47Z).
    """
    if not date_string or date_string == 'N/A':
        return 'N/A'
    
    try:
        # Parse the date string (handles both with and without microseconds)
        if '.' in date_string:
            # Old format with microseconds
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        else:
            # Already in correct format or GitHub format
            return date_string
        
        # Convert to standardized format
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


# =============================================================================
# GITHUB API OPERATIONS
# =============================================================================

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.

    Returns the final requests.Response on success or non-retryable status, or None after exhausting retries.
    """
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers or {},
                params=params,
                json=json_body,
                data=data,
                timeout=timeout
            )

            status = resp.status_code

            # Success
            if 200 <= status < 300:
                return resp

            # Rate limits or server errors -> retry with backoff
            if status in (403, 429) or 500 <= status < 600:
                wait = None

                # Prefer Retry-After when present
                retry_after = resp.headers.get('Retry-After') or resp.headers.get('retry-after')
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except Exception:
                        wait = None

                # Fallback to X-RateLimit-Reset when 403/429
                if wait is None and status in (403, 429):
                    reset_hdr = resp.headers.get('X-RateLimit-Reset') or resp.headers.get('x-ratelimit-reset')
                    if reset_hdr:
                        try:
                            reset_ts = int(float(reset_hdr))
                            wait = max(reset_ts - time.time() + 2, 1)
                        except Exception:
                            wait = None

                # Final fallback: exponential backoff with jitter
                if wait is None:
                    wait = delay + random.uniform(0, 0.5)

                # Cap individual wait to avoid extreme sleeps
                wait = max(1.0, min(wait, 120.0))
                print(f"GitHub API {status}. Backing off {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue

            # Non-retryable error; return response for caller to handle
            return resp

        except requests.RequestException as e:
            # Network error -> retry with backoff
            wait = delay + random.uniform(0, 0.5)
            wait = max(1.0, min(wait, 60.0))
            print(f"Request error: {e}. Retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            delay = min(delay * 2, 60.0)

    print(f"Exceeded max retries for {url}")
    return None

def get_github_token():
    """Get GitHub token from environment variables."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found. API rate limits: 60/hour (authenticated: 5000/hour)")
    return token


def validate_github_username(identifier):
    """Verify that a GitHub identifier exists with backoff-aware requests."""
    try:
        token = get_github_token()
        headers = {'Authorization': f'token {token}'} if token else {}
        url = f'https://api.github.com/users/{identifier}'
        response = request_with_backoff('GET', url, headers=headers, max_retries=1)
        if response is None:
            return False, "Validation error: network/rate limit exhausted"
        if response.status_code == 200:
            return True, "Username is valid"
        elif response.status_code == 404:
            return False, "GitHub identifier not found"
        else:
            return False, f"Validation error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def fetch_issues_with_time_partition(base_query, start_date, end_date, headers, issues_by_id, debug_limit=None):
    """
    Fetch issues within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.

    Args:
        debug_limit: If set, stops fetching after this many issues (for testing)

    Returns the number of issues found in this time partition.
    """
    # Format dates for GitHub search (YYYY-MM-DD)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    # Add date range to query
    query = f'{base_query} created:{start_str}..{end_str}'

    print(f"  Searching range {start_str} to {end_str}...")

    page = 1
    per_page = 100
    total_in_partition = 0

    while True:
        # Check debug limit
        if debug_limit is not None and total_in_partition >= debug_limit:
            print(f"    🐛 DEBUG MODE: Reached limit of {debug_limit} issues, stopping...")
            return total_in_partition
        url = 'https://api.github.com/search/issues'
        params = {
            'q': query,
            'per_page': per_page,
            'page': page,
            'sort': 'created',
            'order': 'asc'
        }

        try:
            response = request_with_backoff('GET', url, headers=headers, params=params)
            if response is None:
                print(f"    Error: retries exhausted for range {start_str} to {end_str}")
                return total_in_partition

            if response.status_code != 200:
                print(f"    Error: HTTP {response.status_code} for range {start_str} to {end_str}")
                return total_in_partition

            data = response.json()
            total_count = data.get('total_count', 0)
            items = data.get('items', [])

            if not items:
                break

            # Add issues to global dict
            for issue in items:
                issue_id = issue.get('id')
                if issue_id and issue_id not in issues_by_id:
                    issues_by_id[issue_id] = issue
                    total_in_partition += 1

            # Check if we hit the 1000-result limit
            if total_count > 1000 and page == 10:
                print(f"    ⚠️ Hit 1000-result limit ({total_count} total). Splitting time range...")

                # Calculate midpoint
                time_diff = end_date - start_date
                mid_date = start_date + time_diff / 2

                # Recursively fetch both halves
                count1 = fetch_issues_with_time_partition(base_query, start_date, mid_date, headers, issues_by_id, debug_limit)
                count2 = fetch_issues_with_time_partition(base_query, mid_date + timedelta(days=1), end_date, headers, issues_by_id, debug_limit)

                return count1 + count2

            # Normal pagination: check if there are more pages
            if len(items) < per_page or page >= 10:
                break

            page += 1
            time.sleep(0.5)  # Courtesy delay between pages

        except Exception as e:
            print(f"    Error fetching range {start_str} to {end_str}: {str(e)}")
            return total_in_partition

    if total_in_partition > 0:
        print(f"    ✓ Found {total_in_partition} issues in range {start_str} to {end_str}")

    return total_in_partition


def extract_issue_metadata(issue):
    """
    Extract minimal issue metadata for efficient storage.
    Only keeps essential fields: html_url, created_at, closed_at, state_reason.
    Note: agent_name is not stored as it's inferred from the folder structure.

    Issue states:
    - state: "open" or "closed"
    - state_reason: "completed" (resolved), "not_planned" (closed as not planned), or None (still open)
    """
    # Extract dates and state
    created_at = issue.get('created_at')
    closed_at = issue.get('closed_at')
    state = issue.get('state')
    state_reason = issue.get('state_reason')

    return {
        'html_url': issue.get('html_url'),
        'created_at': created_at,
        'closed_at': closed_at,
        'state': state,
        'state_reason': state_reason
    }


def fetch_all_issues_metadata(identifier, agent_name, token=None, start_from_date=None, year=None, exclude_dates=None):
    """
    Fetch issues associated with a GitHub user or bot for the past 6 months.
    Returns lightweight metadata instead of full issue objects.

    This function employs time-based partitioning to navigate GitHub's 1000-result limit per query.
    It searches using multiple query patterns:
    - is:issue author:{identifier} (issues authored by the bot)
    - is:issue assignee:{identifier} (issues assigned to the bot)

    Args:
        identifier: GitHub username or bot identifier
        agent_name: Human-readable name of the agent for metadata purposes
        token: GitHub API token for authentication
        start_from_date: Only fetch issues created after this date (for incremental updates)
        year: Year parameter (deprecated, retained for compatibility but not utilized)
        exclude_dates: Set of date objects to exclude from mining (dates that have already been processed)

    Returns:
        List of dictionaries containing minimal issue metadata
    """
    headers = {'Authorization': f'token {token}'} if token else {}

    # Debug mode: limit issue retrieval for testing
    debug_limit_per_pattern = 10 if DEBUG_MODE else None

    if DEBUG_MODE:
        print(f"\n🐛 DEBUG MODE ENABLED: Limiting to {debug_limit_per_pattern} issues per query pattern")

    # Define query patterns for issues:
    # 1) author pattern: issues authored by the identifier
    # 2) assignee pattern: issues assigned to the identifier
    stripped_id = identifier.replace('[bot]', '')
    query_patterns = []

    # Always add author and assignee pattern
    query_patterns.append(f'is:issue author:{identifier}')
    query_patterns.append(f'is:issue assignee:{identifier}')
    query_patterns.append(f'is:issue assignee:{stripped_id}')

    # Use a dict to deduplicate issues by ID
    issues_by_id = {}

    # Define time range: past 6 months only (or from start_from_date if specified)
    current_time = datetime.now(timezone.utc)
    six_months_ago = current_time - timedelta(days=180)  # ~6 months

    if start_from_date:
        # Use start_from_date but ensure it's not older than 6 months
        start_date = max(start_from_date, six_months_ago)
    else:
        start_date = six_months_ago

    # End date is current time
    end_date = current_time

    for query_pattern in query_patterns:
        print(f"\n🔍 Searching with query: {query_pattern}")
        print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        pattern_start_time = time.time()
        initial_count = len(issues_by_id)

        # Fetch with time partitioning
        issues_found = fetch_issues_with_time_partition(
            query_pattern,
            start_date,
            end_date,
            headers,
            issues_by_id,
            debug_limit_per_pattern
        )

        pattern_duration = time.time() - pattern_start_time
        new_issues = len(issues_by_id) - initial_count

        print(f"   ✓ Pattern complete: {new_issues} new issues found ({issues_found} total fetched, {len(issues_by_id) - initial_count - (issues_found - new_issues)} duplicates)")
        print(f"   ⏱️ Time taken: {pattern_duration:.1f} seconds")

        # Delay between different query patterns (shorter in debug mode)
        time.sleep(0.2 if DEBUG_MODE else 1.0)

    # Convert to lightweight metadata
    all_issues = list(issues_by_id.values())

    # Filter out issues from excluded dates if specified
    if exclude_dates:
        filtered_issues = []
        excluded_count = 0
        for issue in all_issues:
            created_at = issue.get('created_at')
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    issue_date = dt.date()
                    if issue_date not in exclude_dates:
                        filtered_issues.append(issue)
                    else:
                        excluded_count += 1
                except Exception:
                    filtered_issues.append(issue)  # Keep issues with unparseable dates
            else:
                filtered_issues.append(issue)  # Keep issues without created_at

        if excluded_count > 0:
            print(f"   ⏭️ Skipped {excluded_count} issues from already-mined dates")
        all_issues = filtered_issues

    if DEBUG_MODE:
        print(f"\n✅ COMPLETE (DEBUG MODE): Found {len(all_issues)} unique issues for {identifier}")
        print(f"   Note: In production mode, this would fetch ALL issues")
    else:
        print(f"\n✅ COMPLETE: Found {len(all_issues)} unique issues for {identifier}")
    print(f"📦 Extracting minimal metadata...")

    metadata_list = [extract_issue_metadata(issue) for issue in all_issues]

    # Calculate memory savings
    import sys
    original_size = sys.getsizeof(str(all_issues))
    metadata_size = sys.getsizeof(str(metadata_list))
    savings_pct = ((original_size - metadata_size) / original_size * 100) if original_size > 0 else 0

    print(f"💾 Memory efficiency: {original_size // 1024}KB → {metadata_size // 1024}KB (saved {savings_pct:.1f}%)")

    return metadata_list


def calculate_issue_stats_from_metadata(metadata_list):
    """
    Calculate statistics from a list of issue metadata (lightweight objects).
    Works with minimal metadata: html_url, created_at, closed_at, state, state_reason.

    Returns a dictionary with comprehensive issue metrics.

    Resolved Rate is calculated as:
        resolved issues / total issues * 100

    Resolved Issues = issues closed as completed (state_reason="completed")
    We do NOT count issues closed as not planned (state_reason="not_planned")
    """
    total_issues = len(metadata_list)

    # Count resolved issues - those with state_reason="completed"
    resolved = sum(1 for issue_meta in metadata_list
                  if issue_meta.get('state_reason') == 'completed')

    # Calculate resolved rate
    resolved_rate = (resolved / total_issues * 100) if total_issues > 0 else 0

    return {
        'total_issues': total_issues,
        'resolved': resolved,
        'resolved_rate': round(resolved_rate, 2),
    }


def calculate_monthly_metrics_by_agent():
    """
    Calculate monthly metrics for all agents for visualization.
    Loads data directly from SWE-Arena/issue_metadata dataset for the current year.

    Returns:
        dict: {
            'agents': list of agent names,
            'months': list of month labels (e.g., '2025-01'),
            'data': {
                agent_name: {
                    'resolved_rates': list of resolved rates by month,
                    'total_issues': list of issue counts by month,
                    'resolved_issues': list of resolved issue counts by month
                }
            }
        }
    """
    # Get current year for loading metadata
    current_year = datetime.now().year

    # Load ALL agents from HuggingFace agents repo
    agents = load_agents_from_hf()

    # Create mapping from agent_identifier to agent_name
    identifier_to_name = {agent.get('github_identifier'): agent.get('agent_name') for agent in agents if agent.get('github_identifier')}

    # Load all issue metadata for current year from issue_metadata dataset
    all_metadata = load_issue_metadata_for_year(current_year)

    if not all_metadata:
        return {'agents': [], 'months': [], 'data': {}}

    # Group by agent and month
    agent_month_data = defaultdict(lambda: defaultdict(list))

    for issue_meta in all_metadata:
        agent_identifier = issue_meta.get('agent_identifier')
        created_at = issue_meta.get('created_at')

        if not agent_identifier or not created_at:
            continue

        # Get agent_name from identifier
        agent_name = identifier_to_name.get(agent_identifier, agent_identifier)

        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            month_key = f"{dt.year}-{dt.month:02d}"
            agent_month_data[agent_name][month_key].append(issue_meta)
        except Exception as e:
            print(f"Warning: Could not parse date '{created_at}': {e}")
            continue

    # Get all unique months and sort them
    all_months = set()
    for agent_data in agent_month_data.values():
        all_months.update(agent_data.keys())
    months = sorted(list(all_months))

    # Calculate metrics for each agent and month
    result_data = {}
    for agent_name, month_dict in agent_month_data.items():
        resolved_rates = []
        total_issues_list = []
        resolved_issues_list = []

        for month in months:
            issues_in_month = month_dict.get(month, [])

            # Count resolved issues (those with state_reason="completed")
            resolved_count = sum(1 for issue in issues_in_month if issue.get('state_reason') == 'completed')

            # Total issues created in this month
            total_count = len(issues_in_month)

            # Calculate resolved rate
            resolved_rate = (resolved_count / total_count * 100) if total_count > 0 else None

            resolved_rates.append(resolved_rate)
            total_issues_list.append(total_count)
            resolved_issues_list.append(resolved_count)

        result_data[agent_name] = {
            'resolved_rates': resolved_rates,
            'total_issues': total_issues_list,
            'resolved_issues': resolved_issues_list
        }

    return {
        'agents': sorted(list(agent_month_data.keys())),
        'months': months,
        'data': result_data
    }


# =============================================================================
# ISSUE METADATA STORAGE & RETRIEVAL
# =============================================================================

def group_metadata_by_date(metadata_list):
    """
    Group issue metadata by exact date (year.month.day) for efficient daily storage.
    Returns dict: {(year, month, day): [metadata_list]}
    """
    grouped = defaultdict(list)

    for issue_meta in metadata_list:
        created_at = issue_meta.get('created_at')
        if not created_at:
            continue

        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            key = (dt.year, dt.month, dt.day)
            grouped[key].append(issue_meta)
        except Exception as e:
            print(f"Warning: Could not parse date '{created_at}': {e}")

    return dict(grouped)


def save_issue_metadata_to_hf(metadata_list, agent_identifier):
    """
    Save issue metadata to HuggingFace dataset, organized by [agent_identifier]/YYYY.MM.DD.jsonl.
    Each file is stored in the agent's folder and named YYYY.MM.DD.jsonl for that day's issues.
    In debug mode, saves to in-memory cache only.

    This function APPENDS new metadata and DEDUPLICATES by html_url.

    Args:
        metadata_list: List of issue metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    # Skip saving to HF in debug mode - use in-memory cache instead
    if DEBUG_MODE:
        global DEBUG_ISSUE_METADATA_CACHE
        # Merge with existing cache, deduplicating by html_url
        existing = {issue['html_url']: issue for issue in DEBUG_ISSUE_METADATA_CACHE[agent_identifier] if issue.get('html_url')}
        new = {issue['html_url']: issue for issue in metadata_list if issue.get('html_url')}
        existing.update(new)
        DEBUG_ISSUE_METADATA_CACHE[agent_identifier] = list(existing.values())
        print(f"🐛 DEBUG MODE: Saved to in-memory cache only ({len(metadata_list)} issues) - NOT saved to HuggingFace")
        return True

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi()

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        for (issue_year, month, day), day_metadata in grouped.items():
            # New structure: [agent_identifier]/YYYY.MM.DD.jsonl
            filename = f"{agent_identifier}/{issue_year}.{month:02d}.{day:02d}.jsonl"
            local_filename = f"{issue_year}.{month:02d}.{day:02d}.jsonl"
            print(f"📤 Uploading {len(day_metadata)} issues to {filename}...")

            # Download existing file if it exists
            existing_metadata = []
            try:
                file_path = hf_hub_download(
                    repo_id=ISSUE_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                existing_metadata = load_jsonl(file_path)
                print(f"   Found {len(existing_metadata)} existing issues in {filename}")
            except Exception:
                print(f"   No existing file found for {filename}, creating new")

            # Merge and deduplicate by html_url
            existing_by_url = {meta['html_url']: meta for meta in existing_metadata if meta.get('html_url')}
            new_by_url = {meta['html_url']: meta for meta in day_metadata if meta.get('html_url')}

            # Update with new data (new data overwrites old)
            existing_by_url.update(new_by_url)
            merged_metadata = list(existing_by_url.values())

            # Save locally
            save_jsonl(local_filename, merged_metadata)

            try:
                # Upload to HuggingFace with folder path
                upload_with_retry(
                    api=api,
                    path_or_fileobj=local_filename,
                    path_in_repo=filename,
                    repo_id=ISSUE_METADATA_REPO,
                    repo_type="dataset",
                    token=token
                )
                print(f"   ✓ Saved {len(merged_metadata)} total issues to {filename}")
            finally:
                # Always clean up local file, even if upload fails
                if os.path.exists(local_filename):
                    os.remove(local_filename)

        return True

    except Exception as e:
        print(f"✗ Error saving issue metadata: {str(e)}")
        return False


def load_issue_metadata_for_year(year):
    """
    Load all issue metadata for a specific year from HuggingFace.
    Scans all agent folders and loads daily files matching the year.
    In debug mode, loads from in-memory cache if available.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Returns:
        List of dictionaries with 'agent_identifier' added to each issue metadata.
    """
    # In debug mode, check in-memory cache first
    if DEBUG_MODE and DEBUG_ISSUE_METADATA_CACHE:
        all_metadata = []
        for agent_identifier, metadata_list in DEBUG_ISSUE_METADATA_CACHE.items():
            for issue_meta in metadata_list:
                issue_with_agent = issue_meta.copy()
                issue_with_agent['agent_identifier'] = agent_identifier
                all_metadata.append(issue_with_agent)
        if all_metadata:
            print(f"🐛 DEBUG MODE: Loading issue metadata from in-memory cache ({len(all_metadata)} issues)")
            return all_metadata

    try:
        api = HfApi()
        token = get_hf_token()

        # List all files in the repository
        files = api.list_repo_files(repo_id=ISSUE_METADATA_REPO, repo_type="dataset")

        # Filter for files matching the year pattern: [agent_identifier]/YYYY.MM.DD.jsonl
        # Extract year from filename
        year_str = str(year)
        year_files = []
        for f in files:
            if f.endswith('.jsonl'):
                parts = f.split('/')
                if len(parts) == 2:  # [agent_identifier]/YYYY.MM.DD.jsonl
                    filename = parts[1]
                    if filename.startswith(year_str + '.'):
                        year_files.append(f)

        print(f"📥 Loading issue metadata for {year} ({len(year_files)} daily files across all agents)...")

        all_metadata = []
        for filename in year_files:
            try:
                # Extract agent_identifier from path (first part)
                # Format: agent_identifier/YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    print(f"   Warning: Unexpected filename format: {filename}")
                    continue

                agent_identifier = parts[0]

                file_path = hf_hub_download(
                    repo_id=ISSUE_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                day_metadata = load_jsonl(file_path)

                # Add agent_identifier to each issue metadata for processing
                for issue_meta in day_metadata:
                    issue_meta['agent_identifier'] = agent_identifier

                all_metadata.extend(day_metadata)
                print(f"   ✓ Loaded {len(day_metadata)} issues from {filename}")
            except Exception as e:
                print(f"   Warning: Could not load {filename}: {str(e)}")

        print(f"✓ Loaded {len(all_metadata)} total issues for {year}")
        return all_metadata

    except Exception as e:
        print(f"✗ Error loading issue metadata for {year}: {str(e)}")
        return []


def get_latest_issue_date_for_agent(agent_identifier):
    """
    Get the latest issue creation date for an agent from stored metadata.
    Used for incremental updates - only fetch issues newer than this date.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Args:
        agent_identifier: GitHub identifier of the agent

    Returns:
        datetime or None if no existing issues found.
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # List all files in the repository
        files = api.list_repo_files(repo_id=ISSUE_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        # New structure: [agent_identifier]/YYYY.MM.DD.jsonl
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        if not agent_files:
            return None

        # Find latest created_at across all files
        latest_date = None
        for filename in agent_files:
            try:
                file_path = hf_hub_download(
                    repo_id=ISSUE_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=token
                )
                metadata = load_jsonl(file_path)

                for issue in metadata:
                    created_at = issue.get('created_at')
                    if created_at:
                        try:
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            if latest_date is None or dt > latest_date:
                                latest_date = dt
                        except Exception:
                            continue
            except Exception:
                continue

        return latest_date

    except Exception:
        return None


def get_daily_files_last_n_months(agent_identifier, n_months=6):
    """
    Get list of daily file paths for an agent from the last N months.

    Args:
        agent_identifier: GitHub identifier of the agent
        n_months: Number of months to look back (default: 6)

    Returns:
        List of file paths in format: [agent_identifier]/YYYY.MM.DD.jsonl
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # Calculate date range
        today = datetime.now(timezone.utc)
        n_months_ago = today - timedelta(days=30 * n_months)

        # List all files in the repository
        files = api.list_repo_files(repo_id=ISSUE_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        # Filter by date range (extract date from filename)
        recent_files = []
        for filename in agent_files:
            try:
                # Extract date from filename: YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    continue

                date_part = parts[1].replace('.jsonl', '')  # Get YYYY.MM.DD
                date_components = date_part.split('.')
                if len(date_components) != 3:
                    continue

                file_year, file_month, file_day = map(int, date_components)
                file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                # Include if within last n_months
                if n_months_ago <= file_date <= today:
                    recent_files.append(filename)
            except Exception:
                continue

        return recent_files

    except Exception as e:
        print(f"Error getting daily files: {str(e)}")
        return []


def get_already_mined_dates(agent_identifier, n_months=6):
    """
    Get set of dates that have already been mined for an agent.

    Args:
        agent_identifier: GitHub identifier of the agent
        n_months: Number of months to look back (default: 6)

    Returns:
        Set of date objects (datetime.date) that already have data files
    """
    try:
        api = HfApi()

        # Calculate date range
        today = datetime.now(timezone.utc)
        n_months_ago = today - timedelta(days=30 * n_months)

        # List all files in the repository
        files = api.list_repo_files(repo_id=ISSUE_METADATA_REPO, repo_type="dataset")

        # Filter for files in this agent's folder
        agent_pattern = f"{agent_identifier}/"
        agent_files = [f for f in files if f.startswith(agent_pattern) and f.endswith('.jsonl')]

        mined_dates = set()
        for filename in agent_files:
            try:
                # Extract date from filename: [agent_identifier]/YYYY.MM.DD.jsonl
                parts = filename.split('/')
                if len(parts) != 2:
                    continue

                date_part = parts[1].replace('.jsonl', '')  # Get YYYY.MM.DD
                date_components = date_part.split('.')
                if len(date_components) != 3:
                    continue

                file_year, file_month, file_day = map(int, date_components)
                file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc).date()

                # Only include dates within the last n_months
                if n_months_ago.date() <= file_date <= today.date():
                    mined_dates.add(file_date)
            except Exception as e:
                print(f"   Warning: Could not parse date from filename {filename}: {e}")
                continue

        return mined_dates

    except Exception as e:
        print(f"   Warning: Could not get already-mined dates for {agent_identifier}: {str(e)}")
        return set()


def fetch_issue_current_status(issue_url, token):
    """
    Fetch the current status of a single issue from GitHub API.

    Args:
        issue_url: Issue HTML URL (e.g., https://github.com/owner/repo/issues/123)
        token: GitHub API token

    Returns:
        Dictionary with updated state, state_reason, and closed_at, or None if failed
    """
    try:
        # Convert HTML URL to API URL
        # https://github.com/owner/repo/issues/123 -> https://api.github.com/repos/owner/repo/issues/123
        parts = issue_url.replace('https://github.com/', '').split('/')
        if len(parts) < 4:
            return None

        owner, repo, issue_word, issue_number = parts[0], parts[1], parts[2], parts[3]
        api_url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}'

        headers = {'Authorization': f'token {token}'} if token else {}
        response = request_with_backoff('GET', api_url, headers=headers, max_retries=3)

        if response is None or response.status_code != 200:
            return None

        issue_data = response.json()
        state = issue_data.get('state')
        state_reason = issue_data.get('state_reason')
        closed_at = issue_data.get('closed_at')

        return {
            'state': state,
            'state_reason': state_reason,
            'closed_at': closed_at
        }

    except Exception as e:
        print(f"   Error fetching issue status for {issue_url}: {str(e)}")
        return None


def refresh_open_issues_for_agent(agent_identifier, token):
    """
    Refresh status for all open issues from the last 6 months for an agent.
    Only updates issues that are still open (state="open" or no state_reason).

    This implements the smart update strategy:
    - Skip issues that are already closed/resolved
    - Fetch current status for open issues
    - Update and save back to daily files

    Args:
        agent_identifier: GitHub identifier of the agent
        token: GitHub API token

    Returns:
        Tuple: (total_checked, updated_count)
    """
    print(f"\n🔄 Refreshing open issues for {agent_identifier} (last 6 months)...")

    try:
        # Get daily files from last 6 months
        recent_files = get_daily_files_last_n_months(agent_identifier, n_months=6)

        if not recent_files:
            print(f"   No recent files found for {agent_identifier}")
            return (0, 0)

        print(f"   Found {len(recent_files)} daily files to check")

        total_checked = 0
        updated_count = 0

        # Process each file
        for filename in recent_files:
            try:
                # Download file
                file_path = hf_hub_download(
                    repo_id=ISSUE_METADATA_REPO,
                    filename=filename,
                    repo_type="dataset",
                    token=get_hf_token()
                )
                issues = load_jsonl(file_path)

                if not issues:
                    continue

                updated_issues = []
                file_had_updates = False

                # Check each issue
                for issue in issues:
                    # Skip if already closed (has a state_reason)
                    if issue.get('state') == 'closed' and issue.get('state_reason'):
                        updated_issues.append(issue)
                        continue

                    # Issue is open, fetch current status
                    total_checked += 1
                    issue_url = issue.get('html_url')

                    if not issue_url:
                        updated_issues.append(issue)
                        continue

                    current_status = fetch_issue_current_status(issue_url, token)

                    if current_status:
                        # Check if status changed (now closed)
                        if current_status['state'] == 'closed':
                            print(f"   ✓ Issue status changed: {issue_url}")
                            issue['state'] = current_status['state']
                            issue['state_reason'] = current_status['state_reason']
                            issue['closed_at'] = current_status['closed_at']
                            updated_count += 1
                            file_had_updates = True

                    updated_issues.append(issue)
                    time.sleep(0.1)  # Rate limiting courtesy delay

                # Save file if there were updates
                if file_had_updates:
                    # Extract filename components for local save
                    parts = filename.split('/')
                    local_filename = parts[-1]  # Just YYYY.MM.DD.jsonl

                    # Save locally
                    save_jsonl(local_filename, updated_issues)

                    try:
                        # Upload back to HuggingFace
                        api = HfApi()
                        upload_with_retry(
                            api=api,
                            path_or_fileobj=local_filename,
                            path_in_repo=filename,
                            repo_id=ISSUE_METADATA_REPO,
                            repo_type="dataset",
                            token=get_hf_token()
                        )
                        print(f"   💾 Updated {filename}")
                    finally:
                        # Always clean up local file, even if upload fails
                        if os.path.exists(local_filename):
                            os.remove(local_filename)

            except Exception as e:
                print(f"   Warning: Could not process {filename}: {str(e)}")
                continue

        print(f"   ✅ Refresh complete: {total_checked} open issues checked, {updated_count} updated")
        return (total_checked, updated_count)

    except Exception as e:
        print(f"   ✗ Error refreshing issues for {agent_identifier}: {str(e)}")
        return (0, 0)


# =============================================================================
# HUGGINGFACE DATASET OPERATIONS
# =============================================================================

def load_agents_from_hf():
    """Load all agent metadata JSON files from HuggingFace dataset."""
    try:
        api = HfApi()
        agents = []

        # List all files in the repository
        files = api.list_repo_files(repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

        print(f"Found {len(json_files)} agent files in {AGENTS_REPO}")

        # Download and parse each JSON file
        for json_file in json_files:
            try:
                file_path = hf_hub_download(
                    repo_id=AGENTS_REPO,
                    filename=json_file,
                    repo_type="dataset"
                )

                with open(file_path, 'r') as f:
                    agent_data = json.load(f)
                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return None


def load_leaderboard_dataset():
    """Load leaderboard data from HuggingFace dataset for current year.
    In debug mode, loads from in-memory cache if available."""
    # In debug mode, check in-memory cache first
    if DEBUG_MODE and DEBUG_LEADERBOARD_CACHE:
        print(f"🐛 DEBUG MODE: Loading leaderboard from in-memory cache ({len(DEBUG_LEADERBOARD_CACHE)} entries)")
        return list(DEBUG_LEADERBOARD_CACHE.values())

    try:
        year = datetime.now().year
        filename = f"{year}.csv"

        # Try to download the CSV file for current year
        file_path = hf_hub_download(
            repo_id=LEADERBOARD_REPO,
            filename=filename,
            repo_type="dataset"
        )

        # Load CSV into list of dicts
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
        print(f"✓ Loaded {len(data)} entries from {filename}")
        return data

    except Exception as e:
        print(f"Could not load leaderboard dataset for year {datetime.now().year}: {str(e)}")
        return None


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.

    Args:
        api: HfApi instance
        path_or_fileobj: Local file path to upload
        path_in_repo: Target path in the repository
        repo_id: Repository ID
        repo_type: Type of repository (e.g., "dataset")
        token: HuggingFace token
        max_retries: Maximum number of retry attempts

    Returns:
        True if upload succeeded, raises exception if all retries failed
    """
    delay = 2.0  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                token=token
            )
            if attempt > 0:
                print(f"   ✓ Upload succeeded on attempt {attempt + 1}/{max_retries}")
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay + random.uniform(0, 1.0)
                print(f"   ⚠️ Upload failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                print(f"   ⏳ Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                delay = min(delay * 2, 60.0)  # Exponential backoff, max 60s
            else:
                print(f"   ✗ Upload failed after {max_retries} attempts: {str(e)}")
                raise


def save_agent_to_hf(data):
    """Save a new agent to HuggingFace dataset as {identifier}.json in root."""
    try:
        api = HfApi()
        token = get_hf_token()

        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        identifier = data['github_identifier']
        filename = f"{identifier}.json"

        # Save locally first
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        try:
            # Upload to HuggingFace (root directory)
            upload_with_retry(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=AGENTS_REPO,
                repo_type="dataset",
                token=token
            )
            print(f"✓ Saved agent to HuggingFace: {filename}")
            return True
        finally:
            # Always clean up local file, even if upload fails
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"✗ Error saving agent: {str(e)}")
        return False


def save_leaderboard_to_hf(cache_dict):
    """Save complete leaderboard to HuggingFace dataset as CSV.
    In debug mode, saves to in-memory cache only."""
    # Skip saving in debug mode - use in-memory cache instead
    if DEBUG_MODE:
        global DEBUG_LEADERBOARD_CACHE
        # Filter out agents with zero total issues
        filtered_cache_dict = {k: v for k, v in cache_dict.items() if v.get('total_issues', 0) > 0}
        DEBUG_LEADERBOARD_CACHE = filtered_cache_dict.copy()
        data_list = dict_to_cache(filtered_cache_dict)
        print(f"🐛 DEBUG MODE: Saved to in-memory cache only ({len(data_list)} entries) - NOT saved to HuggingFace")
        return True

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found. Please set HF_TOKEN in your Space settings.")

        # Filter out agents with zero total issues
        filtered_cache_dict = {k: v for k, v in cache_dict.items() if v.get('total_issues', 0) > 0}
        # Convert to DataFrame
        data_list = dict_to_cache(filtered_cache_dict)
        df = pd.DataFrame(data_list)

        # Save to CSV with year as filename
        year = datetime.now().year
        filename = f"{year}.csv"
        df.to_csv(filename, index=False)

        try:
            # Upload to HuggingFace
            api = HfApi()
            upload_with_retry(
                api=api,
                path_or_fileobj=filename,
                path_in_repo=filename,
                repo_id=LEADERBOARD_REPO,
                repo_type="dataset",
                token=token
            )
            print(f"✓ Saved leaderboard to HuggingFace as {filename} ({len(data_list)} entries)")
            return True
        finally:
            # Always clean up local file, even if upload fails
            if os.path.exists(filename):
                os.remove(filename)

    except Exception as e:
        print(f"✗ Error saving leaderboard: {str(e)}")
        return False


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def update_all_agents_incremental():
    """
    Memory-efficient incremental update of issue statistics for all agents.

    Strategy:
    1. For each agent, load existing data from SWE-Arena/issue_metadata
    2. Identify already-mined dates (based on filename: YYYY.MM.DD.jsonl)
    3. Only fetch issues from dates that haven't been mined yet (within last 6 months)
    4. If no data exists at all, mine everything from scratch
    5. Store minimal metadata (not full issue objects) to avoid storage limits
    6. Construct leaderboard from ALL stored metadata (last 6 months)

    Returns dictionary of all agent data with current stats.
    """
    token = get_github_token()
    current_year = datetime.now().year

    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return {}

    cache_dict = {}

    # Update each agent
    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        if not identifier:
            print(f"Warning: Skipping agent without identifier: {agent}")
            continue

        try:
            print(f"\n{'='*80}")
            print(f"Processing: {agent_name} ({identifier})")
            print(f"{'='*80}")

            # Get already-mined dates for this agent (last 6 months)
            already_mined_dates = get_already_mined_dates(identifier, n_months=6)

            if already_mined_dates:
                print(f"📅 Found {len(already_mined_dates)} already-mined dates")
                print(f"   Skipping these dates and fetching only new data...")
                # Fetch only issues from dates not yet mined
                new_metadata = fetch_all_issues_metadata(
                    identifier,
                    agent_name,
                    token,
                    start_from_date=None,  # Use full 6-month range
                    exclude_dates=already_mined_dates  # But exclude already-mined dates
                )
            else:
                print(f"📅 No existing data found. Mining everything from scratch...")
                # Mine everything from scratch (full 6-month range)
                new_metadata = fetch_all_issues_metadata(
                    identifier,
                    agent_name,
                    token,
                    start_from_date=None
                )

            if new_metadata:
                # Save new metadata to HuggingFace (organized by agent_identifier/YYYY.MM.DD.jsonl)
                print(f"💾 Saving {len(new_metadata)} new issue records...")
                save_issue_metadata_to_hf(new_metadata, identifier)
            else:
                print(f"   No new issues to save")

            # Load ALL metadata for current year to calculate stats (aggregates entire last 6 months)
            print(f"📊 Calculating statistics from ALL stored metadata (last 6 months)...")
            all_year_metadata = load_issue_metadata_for_year(current_year)

            # Filter for this specific agent
            agent_metadata = [issue for issue in all_year_metadata if issue.get('agent_identifier') == identifier]

            # Calculate stats from metadata
            stats = calculate_issue_stats_from_metadata(agent_metadata)

            # Format agent_name as markdown link if website is available
            website = agent.get('website', '')
            formatted_agent_name = f"[{agent_name}]({website})" if website else agent_name

            # Merge metadata with stats
            cache_dict[identifier] = {
                'agent_name': formatted_agent_name,
                'organization': agent.get('organization', 'Unknown'),
                'github_identifier': identifier,
                **stats
            }

            print(f"✓ Updated {identifier}: {stats['total_issues']} issues, {stats['resolved_rate']}% resolved")

        except Exception as e:
            print(f"✗ Error updating {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return cache_dict


def construct_leaderboard_from_metadata():
    """
    Construct leaderboard from stored issue metadata instead of fetching all issues.
    Much more memory-efficient and faster.

    Returns dictionary of agent stats.
    """
    print("📊 Constructing leaderboard from issue metadata...")
    current_year = datetime.now().year

    # Load agents
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found")
        return {}

    # Load all issue metadata for current year
    all_metadata = load_issue_metadata_for_year(current_year)

    cache_dict = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        # Filter metadata for this agent
        agent_metadata = [issue for issue in all_metadata if issue.get('agent_identifier') == identifier]

        # Calculate stats
        stats = calculate_issue_stats_from_metadata(agent_metadata)

        # Format agent_name as markdown link if website is available
        website = agent.get('website', '')
        formatted_agent_name = f"[{agent_name}]({website})" if website else agent_name

        cache_dict[identifier] = {
            'agent_name': formatted_agent_name,
            'organization': agent.get('organization', 'Unknown'),
            'github_identifier': identifier,
            **stats
        }

    return cache_dict


def initialize_data():
    """
    Initialize data on application startup.
    Priority: 1) Leaderboard dataset ({year}.csv), 2) Issue metadata (if available), 3) Full GitHub mining

    In DEBUG MODE:
    - If no data available, automatically mine up to 10 issues per query per agent
    - Does NOT save to HuggingFace datasets
    """
    print("🚀 Initializing leaderboard data...")

    # STEP 1: Try loading existing leaderboard CSV file for current year
    current_year = datetime.now().year
    print(f"📂 Checking for {current_year}.csv in {LEADERBOARD_REPO}...")
    leaderboard_data = load_leaderboard_dataset()
    if leaderboard_data:
        print(f"✓ Initialized from leaderboard dataset ({current_year}.csv)")
        return

    # STEP 2: Try constructing from issue metadata (fast, memory-efficient)
    print(f"📂 {current_year}.csv not found. Checking {ISSUE_METADATA_REPO} for existing data...")
    try:
        cache_dict = construct_leaderboard_from_metadata()
        # Check if there's actually meaningful data (at least one agent with issues)
        has_data = any(entry.get('total_issues', 0) > 0 for entry in cache_dict.values())
        if cache_dict and has_data:
            print(f"✓ Found existing issue metadata. Building leaderboard from {ISSUE_METADATA_REPO}...")
            save_leaderboard_to_hf(cache_dict)
            print("✓ Initialized from issue metadata")
            return
        else:
            print(f"   No meaningful data found in {ISSUE_METADATA_REPO}")
    except Exception as e:
        print(f"   Could not construct from metadata: {e}")

    # If in debug mode and no data available, mine immediately
    if DEBUG_MODE:
        print("\n🐛 DEBUG MODE: No data available, mining immediately (up to 10 issues per query per agent)...")
        agents = load_agents_from_hf()
        if agents:
            print(f"✓ Loaded {len(agents)} agents from HuggingFace")
            print("⛏️ Mining GitHub data in debug mode (limited to 10 issues per query)...")
            cache_dict = update_all_agents_incremental()
            if cache_dict:
                # In debug mode, this won't actually save to HF
                save_leaderboard_to_hf(cache_dict)
                print("✓ Debug mining complete (data NOT saved to HuggingFace)")
            return
        else:
            print("⚠️ No agents found. Waiting for first submission...")
            return

    # Production mode: Fallback to full incremental mining from GitHub
    agents = load_agents_from_hf()
    if agents:
        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        print("⛏️ Mining GitHub data (this may take a while)...")
        cache_dict = update_all_agents_incremental()
        if cache_dict:
            save_leaderboard_to_hf(cache_dict)
        return

    # No data available
    print("⚠️ No data sources available. Waiting for first submission...")


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def create_monthly_metrics_plot():
    """
    Create a Plotly figure with dual y-axes showing:
    - Left y-axis: Resolved Rate (%) as line curves
    - Right y-axis: Total Issues created as bar charts

    Each agent gets a unique color for both their line and bars.
    """
    metrics = calculate_monthly_metrics_by_agent()

    if not metrics['agents'] or not metrics['months']:
        # Return an empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=None,
            xaxis_title=None,
            height=500
        )
        return fig

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Define colors for agents (using a color palette)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    agents = metrics['agents']
    months = metrics['months']
    data = metrics['data']

    # Add traces for each agent
    for idx, agent_name in enumerate(agents):
        color = colors[idx % len(colors)]
        agent_data = data[agent_name]

        # Add line trace for resolved rate (left y-axis)
        resolved_rates = agent_data['resolved_rates']
        # Filter out None values for plotting
        x_resolved = [month for month, rate in zip(months, resolved_rates) if rate is not None]
        y_resolved = [rate for rate in resolved_rates if rate is not None]

        if x_resolved and y_resolved:  # Only add trace if there's data
            fig.add_trace(
                go.Scatter(
                    x=x_resolved,
                    y=y_resolved,
                    name=agent_name,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                    legendgroup=agent_name,
                    showlegend=True,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Resolved Rate: %{y:.2f}%<br>' +
                                 '<extra></extra>'
                ),
                secondary_y=False
            )

        # Add bar trace for total issues (right y-axis)
        # Only show bars for months where agent has issues
        x_bars = []
        y_bars = []
        for month, count in zip(months, agent_data['total_issues']):
            if count > 0:  # Only include months with issues
                x_bars.append(month)
                y_bars.append(count)

        if x_bars and y_bars:  # Only add trace if there's data
            fig.add_trace(
                go.Bar(
                    x=x_bars,
                    y=y_bars,
                    name=f"{agent_name} (Issues)",
                    marker=dict(color=color, opacity=0.6),
                    legendgroup=agent_name,
                    showlegend=False,  # Don't show in legend (already shown for line)
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                 'Month: %{x}<br>' +
                                 'Total Issues: %{y}<br>' +
                                 '<extra></extra>',
                    offsetgroup=agent_name  # Group bars by agent for proper spacing
                ),
                secondary_y=True
            )

    # Update axes labels
    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text="<b>Resolved Rate (%)</b>", secondary_y=False)
    fig.update_yaxes(title_text="<b>Total Issues</b>", secondary_y=True)

    # Update layout
    fig.update_layout(
        title=None,
        hovermode='x unified',
        barmode='group',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=100, b=50)
    )

    return fig


def get_leaderboard_dataframe():
    """
    Load leaderboard data from HuggingFace and convert to pandas DataFrame for display.
    Returns formatted DataFrame sorted by acceptance rate.
    """
    # Load leaderboard data from HuggingFace
    leaderboard_data = load_leaderboard_dataset()

    if not leaderboard_data:
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    for data in leaderboard_data:
        # Filter out agents with zero total issues
        if data.get('total_issues', 0) == 0:
            continue
        # Only include display-relevant fields
        rows.append([
            data.get('agent_name', 'Unknown'),
            data.get('organization', 'Unknown'),
            data.get('total_issues', 0),
            data.get('resolved', 0),
            data.get('resolved_rate', 0.0),
        ])

    # Create DataFrame
    column_names = [col[0] for col in LEADERBOARD_COLUMNS]
    df = pd.DataFrame(rows, columns=column_names)

    # Ensure numeric types
    numeric_cols = ["Total Issues", "Resolved Issues", "Resolved Rate (%)"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Sort by Resolved Rate (%) descending
    if "Resolved Rate (%)" in df.columns and not df.empty:
        df = df.sort_values(by="Resolved Rate (%)", ascending=False).reset_index(drop=True)

    return df


def submit_agent(identifier, agent_name, organization, description, website):
    """
    Submit a new agent to the leaderboard.
    Validates input, saves submission, and fetches PR metadata (memory-efficient).
    """
    # Validate required fields
    if not identifier or not identifier.strip():
        return "❌ GitHub identifier is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not agent_name or not agent_name.strip():
        return "❌ Agent name is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not organization or not organization.strip():
        return "❌ Organization name is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not website or not website.strip():
        return "❌ Website URL is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Clean inputs
    identifier = identifier.strip()
    agent_name = agent_name.strip()
    organization = organization.strip()
    description = description.strip()
    website = website.strip()

    # Validate GitHub identifier
    is_valid, message = validate_github_username(identifier)
    if not is_valid:
        return f"❌ {message}", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Check for duplicates by loading agents from HuggingFace
    agents = load_agents_from_hf()
    if agents:
        existing_names = {agent['github_identifier'] for agent in agents}
        if identifier in existing_names:
            return f"⚠️ Agent with identifier '{identifier}' already exists", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Create submission
    submission = {
        'agent_name': agent_name,
        'organization': organization,
        'github_identifier': identifier,
        'description': description,
        'website': website,
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "❌ Failed to save submission", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Fetch issue metadata immediately (memory-efficient)
    token = get_github_token()
    try:
        print(f"Fetching issue metadata for {agent_name}...")

        # Fetch lightweight metadata
        metadata_list = fetch_all_issues_metadata(identifier, agent_name, token)

        if metadata_list:
            # Save metadata to HuggingFace
            save_issue_metadata_to_hf(metadata_list, identifier)

        # Calculate stats from metadata
        stats = calculate_issue_stats_from_metadata(metadata_list)

        # Format agent_name as markdown link
        formatted_agent_name = f"[{agent_name}]({website})" if website else agent_name

        # Load current leaderboard
        leaderboard_data = load_leaderboard_dataset()
        if not leaderboard_data:
            leaderboard_data = []

        # Convert to dict for easy updating
        cache_dict = {entry['github_identifier']: entry for entry in leaderboard_data}
        # Create submission with formatted agent name for leaderboard
        leaderboard_entry = {**submission, 'agent_name': formatted_agent_name, **stats}
        cache_dict[identifier] = leaderboard_entry

        # Save to HuggingFace
        save_leaderboard_to_hf(cache_dict)

        return f"✅ Successfully submitted {agent_name}!", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    except Exception as e:
        error_msg = f"⚠️ Submitted {agent_name}, but failed to fetch issue data: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return error_msg, get_leaderboard_dataframe(), create_monthly_metrics_plot()


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

def daily_update_task():
    """
    Daily scheduled task (runs at 12:00 AM UTC) for smart issue updates.

    Strategy:
    1. For each agent, refresh open issues from last 6 months
    2. Skip issues that are already closed/resolved (no API calls)
    3. Only fetch status for open issues to check if they've been closed/resolved
    4. Update leaderboard with refreshed data

    This is much more efficient than fetching all issues every time.
    """
    print(f"\n{'='*80}")
    print(f"🕛 Daily update started at {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*80}")

    try:
        token = get_github_token()

        # Load all agents
        agents = load_agents_from_hf()
        if not agents:
            print("No agents found")
            return

        print(f"📋 Processing {len(agents)} agents...")

        total_checked = 0
        total_updated = 0

        # Refresh open issues for each agent (last 6 months)
        for agent in agents:
            identifier = agent.get('github_identifier')
            agent_name = agent.get('agent_name', 'Unknown')

            if not identifier:
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {agent_name} ({identifier})")
            print(f"{'='*60}")

            # Refresh open issues from last 6 months
            checked, updated = refresh_open_issues_for_agent(identifier, token)
            total_checked += checked
            total_updated += updated

        print(f"\n{'='*80}")
        print(f"📊 Refresh Summary:")
        print(f"   Total open issues checked: {total_checked}")
        print(f"   Issues updated (closed/resolved): {total_updated}")
        print(f"{'='*80}")

        # Reconstruct leaderboard from all stored metadata
        print(f"\n📈 Rebuilding leaderboard from refreshed data...")
        cache_dict = construct_leaderboard_from_metadata()

        if cache_dict:
            # Save leaderboard
            save_leaderboard_to_hf(cache_dict)
            print("✓ Leaderboard updated successfully")

        print(f"\n✅ Daily update completed at {datetime.now(timezone.utc).isoformat()}")

    except Exception as e:
        print(f"✗ Daily update failed: {str(e)}")
        import traceback
        traceback.print_exc()


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

# Initialize data before creating UI
if DEBUG_MODE:
    print("\n" + "="*80)
    print("🐛 DEBUG MODE ENABLED 🐛")
    print("="*80)
    print("Issue retrieval is limited to 10 issues per query pattern per agent")

    # Show how debug mode was enabled
    if args.debug:
        print("Enabled via: command-line flag '--debug'")
        print("To disable: run without '--debug' flag")
    else:
        print("Enabled via: DEBUG_MODE environment variable")
        print("To disable: run with '--no-debug' flag or unset DEBUG_MODE")

    print("="*80 + "\n")
else:
    print("\n🚀 Starting in PRODUCTION MODE - full issue retrieval enabled")
    if args.no_debug:
        print("   (Explicitly set via '--no-debug' flag)")
    print()

initialize_data()

# Start APScheduler for daily updates at 12:00 AM UTC
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    daily_update_task,
    trigger=CronTrigger(hour=0, minute=0),  # 12:00 AM UTC daily
    id='daily_issue_refresh',
    name='Daily Issue Status Refresh',
    replace_existing=True
)
scheduler.start()
print("✓ Scheduler started: Daily updates at 12:00 AM UTC")

# Create Gradio interface
with gr.Blocks(title="SWE Agent Issue Leaderboard", theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🏆 SWE Agent Issue Leaderboard")
    gr.Markdown("Track and compare GitHub issue resolution statistics for SWE agents")
    
    with gr.Tabs():
        
        # Leaderboard Tab
        with gr.Tab("📊 Leaderboard"):
            leaderboard_table = Leaderboard(
                value=get_leaderboard_dataframe(),
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Organization"],
                filter_columns=["Resolved Rate (%)"]
            )

            gr.Markdown("### Monthly Metrics")
            gr.Markdown("Track resolution rates and issue activity over time")

            monthly_plot = gr.Plot(
                value=create_monthly_metrics_plot(),
                label="Monthly Issue Metrics"
            )

        # Submit Agent Tab
        with gr.Tab("➕ Submit Agent"):
            
            gr.Markdown("### Submit Your Agent")
            gr.Markdown("Fill in the details below to add your agent to the leaderboard. Make sure you're logged in to HuggingFace CLI on your machine.")
            
            with gr.Row():
                with gr.Column():
                    github_input = gr.Textbox(
                        label="GitHub Identifier*",
                        placeholder="Your agent username (e.g., my-agent-bot)"
                    )
                    name_input = gr.Textbox(
                        label="Agent Name*",
                        placeholder="Your agent's display name"
                    )
                
                with gr.Column():
                    organization_input = gr.Textbox(
                        label="Organization*",
                        placeholder="Your organization or team name"
                    )
                    description_input = gr.Textbox(
                        label="Description",
                        placeholder="Brief description of your agent",
                        lines=3
                    )
                    website_input = gr.Textbox(
                        label="Website",
                        placeholder="https://your-agent-website.com"
                    )
            
            submit_button = gr.Button(
                "Submit Agent",
                variant="primary"
            )
            submission_status = gr.Textbox(
                label="Submission Status",
                interactive=False
            )
            
            # Event handler
            submit_button.click(
                fn=submit_agent,
                inputs=[github_input, name_input, organization_input, description_input, website_input],
                outputs=[submission_status, leaderboard_table, monthly_plot]
            )


# Launch application
if __name__ == "__main__":
    app.launch()