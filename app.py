import gradio as gr
from gradio_leaderboard import Leaderboard
import json
import os
import time
import tempfile
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
import pandas as pd
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/swe_agents"  # HuggingFace dataset for agent metadata
ISSUE_METADATA_REPO = "SWE-Arena/issue_metadata"  # HuggingFace dataset for issue metadata
LEADERBOARD_TIME_FRAME_DAYS = 180  # Time frame for leaderboard
UPDATE_TIME_FRAME_DAYS = 30  # How often to re-mine data via BigQuery

LEADERBOARD_COLUMNS = [
    ("Agent Name", "string"),
    ("Website", "string"),
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
    Also handles space separator (2025-06-23 07:18:28) and incomplete timezone offsets (+00).
    """
    if not date_string or date_string == 'N/A':
        return 'N/A'

    try:
        # Replace space with 'T' for ISO format compatibility
        date_string = date_string.replace(' ', 'T')

        # Fix incomplete timezone offset (+00 or -00 -> +00:00 or -00:00)
        if date_string[-3:-2] in ('+', '-') and ':' not in date_string[-3:]:
            date_string = date_string + ':00'

        # Parse the date string (handles both with and without microseconds)
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))

        # Convert to standardized format
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    except Exception as e:
        print(f"Warning: Could not parse date '{date_string}': {e}")
        return date_string


# =============================================================================
# BIGQUERY OPERATIONS
# =============================================================================

def get_bigquery_client():
    """
    Initialize BigQuery client using credentials from environment variable.

    Expects GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable containing
    the service account JSON credentials as a string.
    """
    # Get the JSON content from environment variable
    creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')

    if creds_json:
        # Create a temporary file to store credentials
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            temp_file.write(creds_json)
            temp_path = temp_file.name

        # Set environment variable to point to temp file
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path

        # Initialize BigQuery client
        client = bigquery.Client()

        # Clean up temp file
        os.unlink(temp_path)

        return client
    else:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")


def generate_table_union_statements(start_date, end_date):
    """
    Generate UNION ALL statements for githubarchive.day tables in date range.

    Args:
        start_date: Start datetime
        end_date: End datetime

    Returns:
        String with UNION ALL SELECT statements for all tables in range
    """
    table_names = []
    current_date = start_date

    while current_date < end_date:
        table_name = f"`githubarchive.day.{current_date.strftime('%Y%m%d')}`"
        table_names.append(table_name)
        current_date += timedelta(days=1)

    # Create UNION ALL chain
    union_parts = [f"SELECT * FROM {table}" for table in table_names]
    return " UNION ALL ".join(union_parts)


def fetch_all_issue_metadata_single_query(client, identifiers, start_date, end_date):
    """
    Fetch issue metadata for ALL agents using ONE comprehensive BigQuery query.

    This query fetches IssuesEvent and IssueCommentEvent from GitHub Archive and
    deduplicates to get the latest state of each issue. Filters by issue author,
    commenter, or assignee.

    Args:
        client: BigQuery client instance
        identifiers: List of GitHub usernames/bot identifiers
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)

    Returns:
        Dictionary mapping agent identifier to list of issue metadata:
        {
            'agent-identifier': [
                {
                    'url': Issue URL,
                    'created_at': Issue creation timestamp,
                    'closed_at': Close timestamp (if closed, else None),
                    'state_reason': Reason for closure (completed/not_planned/etc.)
                },
                ...
            ],
            ...
        }
    """
    print(f"\n🔍 Querying BigQuery for ALL {len(identifiers)} agents in ONE QUERY")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Generate table UNION statements for issue events
    issue_tables = generate_table_union_statements(start_date, end_date)

    # Build identifier list for IN clause (handle both bot and non-bot versions)
    identifier_set = set()
    for id in identifiers:
        identifier_set.add(id)
        # Also add stripped version without [bot] suffix
        stripped = id.replace('[bot]', '')
        if stripped != id:
            identifier_set.add(stripped)

    identifier_list = ', '.join([f"'{id}'" for id in identifier_set])

    # Build comprehensive query with CTEs
    query = f"""
    WITH issue_events AS (
      -- Get all issue events and comment events for ALL agents
      SELECT
        JSON_EXTRACT_SCALAR(payload, '$.issue.html_url') as url,
        JSON_EXTRACT_SCALAR(payload, '$.issue.created_at') as created_at,
        JSON_EXTRACT_SCALAR(payload, '$.issue.closed_at') as closed_at,
        JSON_EXTRACT_SCALAR(payload, '$.issue.state_reason') as state_reason,
        JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') as author,
        JSON_EXTRACT_SCALAR(payload, '$.issue.assignee.login') as assignee,
        JSON_EXTRACT_SCALAR(payload, '$.comment.user.login') as commenter,
        JSON_EXTRACT_SCALAR(payload, '$.issue.number') as issue_number,
        repo.name as repo_name,
        created_at as event_time
      FROM (
        {issue_tables}
      )
      WHERE
        type IN ('IssuesEvent', 'IssueCommentEvent')
        -- Exclude pull requests (they have pull_request field)
        AND JSON_EXTRACT(payload, '$.issue.pull_request') IS NULL
        AND JSON_EXTRACT_SCALAR(payload, '$.issue.html_url') IS NOT NULL
        -- Filter by author OR commenter OR assignee
        AND (
          JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') IN ({identifier_list})
          OR JSON_EXTRACT_SCALAR(payload, '$.comment.user.login') IN ({identifier_list})
          OR JSON_EXTRACT_SCALAR(payload, '$.issue.assignee.login') IN ({identifier_list})
        )
    ),

    latest_states AS (
      -- Deduplicate to get latest state for each issue
      SELECT
        url,
        created_at,
        closed_at,
        state_reason,
        author,
        assignee,
        commenter
      FROM issue_events
      QUALIFY ROW_NUMBER() OVER (
        PARTITION BY repo_name, issue_number
        ORDER BY event_time DESC
      ) = 1
    ),

    agent_issues AS (
      -- Map each issue to its relevant agent(s)
      SELECT DISTINCT
        CASE
          WHEN author IN ({identifier_list}) THEN author
          WHEN commenter IN ({identifier_list}) THEN commenter
          WHEN assignee IN ({identifier_list}) THEN assignee
          ELSE NULL
        END as agent_identifier,
        url,
        created_at,
        closed_at,
        state_reason
      FROM latest_states
      WHERE
        author IN ({identifier_list})
        OR commenter IN ({identifier_list})
        OR assignee IN ({identifier_list})
    )

    SELECT
      agent_identifier,
      url,
      created_at,
      closed_at,
      state_reason
    FROM agent_issues
    WHERE agent_identifier IS NOT NULL
    ORDER BY agent_identifier, created_at DESC
    """

    # Calculate number of days for reporting
    query_days = (end_date - start_date).days

    print(f"   Querying {query_days} days for issue and comment events...")
    print(f"   Agents: {', '.join(identifiers[:5])}{'...' if len(identifiers) > 5 else ''}")

    try:
        query_job = client.query(query)
        results = list(query_job.result())

        print(f"   ✓ Found {len(results)} total issue records across all agents")

        # Group results by agent
        metadata_by_agent = defaultdict(list)

        for row in results:
            agent_id = row.agent_identifier

            # Convert datetime objects to ISO strings
            created_at = row.created_at
            if hasattr(created_at, 'isoformat'):
                created_at = created_at.isoformat()

            closed_at = row.closed_at
            if hasattr(closed_at, 'isoformat'):
                closed_at = closed_at.isoformat()

            metadata_by_agent[agent_id].append({
                'url': row.url,
                'created_at': created_at,
                'closed_at': closed_at,
                'state_reason': row.state_reason,
            })

        # Print breakdown by agent
        print(f"\n   📊 Results breakdown by agent:")
        for identifier in identifiers:
            # Check both original and stripped versions
            count = len(metadata_by_agent.get(identifier, []))
            stripped = identifier.replace('[bot]', '')
            if stripped != identifier:
                count += len(metadata_by_agent.get(stripped, []))

            if count > 0:
                # Merge both versions if needed
                all_metadata = metadata_by_agent.get(identifier, []) + metadata_by_agent.get(stripped, [])
                completed_count = sum(1 for m in all_metadata if m['state_reason'] == 'completed')
                closed_count = sum(1 for m in all_metadata if m['closed_at'] is not None)
                open_count = count - closed_count
                print(f"      {identifier}: {count} issues ({completed_count} completed, {closed_count} closed, {open_count} open)")

        # Convert defaultdict to regular dict and merge bot/non-bot versions
        final_metadata = {}
        for identifier in identifiers:
            combined = metadata_by_agent.get(identifier, [])
            stripped = identifier.replace('[bot]', '')
            if stripped != identifier and stripped in metadata_by_agent:
                combined.extend(metadata_by_agent[stripped])

            if combined:
                final_metadata[identifier] = combined

        return final_metadata

    except Exception as e:
        print(f"   ✗ BigQuery error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}


# =============================================================================
# GITHUB API OPERATIONS (Minimal - for validation only)
# =============================================================================

def get_github_token():
    """Get GitHub token from environment variables for validation purposes."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found for validation")
    return token


def validate_github_username(identifier):
    """Verify that a GitHub identifier exists (simple validation for submission)."""
    try:
        token = get_github_token()
        headers = {'Authorization': f'token {token}'} if token else {}
        url = f'https://api.github.com/users/{identifier}'
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            return True, "Username is valid"
        elif response.status_code == 404:
            return False, "GitHub identifier not found"
        else:
            return False, f"Validation error: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# =============================================================================
# ISSUE METADATA OPERATIONS
# =============================================================================


def extract_issue_metadata(issue):
    """
    Extract minimal issue metadata for efficient storage.
    Only keeps essential fields: url, created_at, closed_at, state_reason.
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
        'url': issue.get('url'),
        'created_at': created_at,
        'closed_at': closed_at,
        'state': state,
        'state_reason': state_reason
    }




def calculate_issue_stats_from_metadata(metadata_list):
    """
    Calculate statistics from a list of issue metadata (lightweight objects).
    Works with minimal metadata: url, created_at, closed_at, state, state_reason.

    Returns a dictionary with comprehensive issue metrics.

    Resolved Rate is calculated as:
        completed issues / closed issues * 100

    Completed Issues = issues closed as completed (state_reason="completed")
    Closed Issues = all issues that have been closed (closed_at is not None)
    We do NOT count issues closed as not planned (state_reason="not_planned") as resolved,
    but they ARE counted in the denominator as closed issues.
    """
    total_issues = len(metadata_list)

    # Count closed issues (those with closed_at timestamp)
    closed_issues = sum(1 for issue_meta in metadata_list
                       if issue_meta.get('closed_at') is not None)

    # Count completed issues (subset of closed issues with state_reason="completed")
    completed = sum(1 for issue_meta in metadata_list
                   if issue_meta.get('state_reason') == 'completed')

    # Calculate resolved rate as: completed / closed (not completed / total)
    resolved_rate = (completed / closed_issues * 100) if closed_issues > 0 else 0

    return {
        'total_issues': total_issues,
        'closed_issues': closed_issues,
        'resolved_issues': completed,
        'resolved_rate': round(resolved_rate, 2),
    }


def calculate_monthly_metrics_by_agent(top_n=None):
    """
    Calculate monthly metrics for all agents (or top N agents) for visualization.
    Loads data directly from SWE-Arena/issue_metadata dataset.

    Args:
        top_n: If specified, only return metrics for the top N agents by total issues.
               Agents are ranked by their total issue count across all months.

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
    # Load ALL agents from HuggingFace agents repo
    agents = load_agents_from_hf()

    # Create mapping from agent_identifier to agent_name
    identifier_to_name = {agent.get('github_identifier'): agent.get('agent_name') for agent in agents if agent.get('github_identifier')}

    # Load all issue metadata from issue_metadata dataset
    all_metadata = load_issue_metadata()

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

            # Count completed issues (those with state_reason="completed")
            completed_count = sum(1 for issue in issues_in_month if issue.get('state_reason') == 'completed')

            # Count closed issues (those with closed_at timestamp)
            closed_count = sum(1 for issue in issues_in_month if issue.get('closed_at') is not None)

            # Total issues created in this month
            total_count = len(issues_in_month)

            # Calculate resolved rate as: completed / closed (not completed / total)
            resolved_rate = (completed_count / closed_count * 100) if closed_count > 0 else None

            resolved_rates.append(resolved_rate)
            total_issues_list.append(total_count)
            resolved_issues_list.append(completed_count)

        result_data[agent_name] = {
            'resolved_rates': resolved_rates,
            'total_issues': total_issues_list,
            'resolved_issues': resolved_issues_list
        }

    # Filter to top N agents if specified
    agents_list = sorted(list(agent_month_data.keys()))
    if top_n is not None and top_n > 0:
        # Calculate total issues for each agent across all months
        agent_totals = []
        for agent_name in agents_list:
            total_issues = sum(result_data[agent_name]['total_issues'])
            agent_totals.append((agent_name, total_issues))

        # Sort by total issues (descending) and take top N
        agent_totals.sort(key=lambda x: x[1], reverse=True)
        top_agents = [agent_name for agent_name, _ in agent_totals[:top_n]]

        # Filter result_data to only include top agents
        result_data = {agent: result_data[agent] for agent in top_agents if agent in result_data}
        agents_list = top_agents

    return {
        'agents': agents_list,
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

    This function uses COMPLETE OVERWRITE strategy (not append/deduplicate).
    Uses upload_large_folder for optimized batch uploads.

    Args:
        metadata_list: List of issue metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    import tempfile
    import shutil

    temp_dir = None
    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        if not grouped:
            print(f"   No valid metadata to save for {agent_identifier}")
            return False

        # Create temporary directory for batch upload
        temp_dir = tempfile.mkdtemp()
        agent_folder = os.path.join(temp_dir, agent_identifier)
        os.makedirs(agent_folder, exist_ok=True)

        print(f"📦 Preparing batch upload for {agent_identifier} ({len(grouped)} daily files)...")

        # Process each daily file
        for (issue_year, month, day), day_metadata in grouped.items():
            filename = f"{agent_identifier}/{issue_year}.{month:02d}.{day:02d}.jsonl"
            local_filename = os.path.join(agent_folder, f"{issue_year}.{month:02d}.{day:02d}.jsonl")

            # Sort by created_at for better organization
            day_metadata.sort(key=lambda x: x.get('created_at', ''), reverse=True)

            # Save to temp directory (complete overwrite, no merging)
            save_jsonl(local_filename, day_metadata)
            print(f"   Prepared {len(day_metadata)} issues for {filename}")

        # Upload entire folder using upload_large_folder (optimized for large files)
        # Note: upload_large_folder creates multiple commits automatically and doesn't support custom commit_message
        print(f"🤗 Uploading {len(grouped)} files ({len(metadata_list)} total issues)...")
        api.upload_large_folder(
            folder_path=temp_dir,
            repo_id=ISSUE_METADATA_REPO,
            repo_type="dataset"
        )
        print(f"   ✓ Batch upload complete for {agent_identifier}")

        return True

    except Exception as e:
        print(f"✗ Error saving issue metadata: {str(e)}")
        return False
    finally:
        # Always clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def load_issue_metadata():
    """
    Load issue metadata from the last LEADERBOARD_TIME_FRAME_DAYS only.
   
    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Returns:
        List of dictionaries with 'agent_identifier' added to each issue metadata.
        Only includes issues within the last LEADERBOARD_TIME_FRAME_DAYS.
    """
    # Calculate cutoff date based on LEADERBOARD_TIME_FRAME_DAYS
    current_time = datetime.now(timezone.utc)
    cutoff_date = current_time - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    try:
        api = HfApi()
        token = get_hf_token()

        # List all files in the repository
        files = api.list_repo_files(repo_id=ISSUE_METADATA_REPO, repo_type="dataset")

        # Filter for files within the time frame: [agent_identifier]/YYYY.MM.DD.jsonl
        # Parse date from filename and only include files within LEADERBOARD_TIME_FRAME_DAYS
        time_frame_files = []
        for f in files:
            if f.endswith('.jsonl'):
                parts = f.split('/')
                if len(parts) == 2:  # [agent_identifier]/YYYY.MM.DD.jsonl
                    filename = parts[1]
                    try:
                        # Extract date from filename: YYYY.MM.DD.jsonl
                        date_part = filename.replace('.jsonl', '')  # Get YYYY.MM.DD
                        date_components = date_part.split('.')
                        if len(date_components) == 3:
                            file_year, file_month, file_day = map(int, date_components)
                            file_date = datetime(file_year, file_month, file_day, tzinfo=timezone.utc)

                            # Only include files within the time frame
                            if file_date >= cutoff_date:
                                time_frame_files.append(f)
                    except Exception:
                        # Skip files with unparseable dates
                        continue

        print(f"📥 [LOAD] Reading cached issue metadata from HuggingFace ({len(time_frame_files)} files, last {LEADERBOARD_TIME_FRAME_DAYS} days)...")

        all_metadata = []
        for filename in time_frame_files:
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

                # Add agent_identifier and filter by date as a double-check
                for issue_meta in day_metadata:
                    # Validate issue date against cutoff
                    created_at = issue_meta.get('created_at')
                    if created_at:
                        try:
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            if dt < cutoff_date:
                                continue  # Skip issues outside time frame
                        except Exception:
                            pass  # Keep issues with unparseable dates

                    issue_meta['agent_identifier'] = agent_identifier
                    all_metadata.append(issue_meta)

                print(f"   ✓ Loaded {len(day_metadata)} issues from {filename}")
            except Exception as e:
                print(f"   Warning: Could not load {filename}: {str(e)}")

        print(f"✓ Loaded {len(all_metadata)} total issues from last {LEADERBOARD_TIME_FRAME_DAYS} days")
        return all_metadata

    except Exception as e:
        print(f"✗ Error loading issue metadata from last {LEADERBOARD_TIME_FRAME_DAYS} days: {str(e)}")
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


def get_daily_files_last_time_frame(agent_identifier):
    """
    Get list of daily file paths for an agent from the configured time frame.

    Args:
        agent_identifier: GitHub identifier of the agent

    Returns:
        List of file paths in format: [agent_identifier]/YYYY.MM.DD.jsonl
    """
    try:
        api = HfApi()
        token = get_hf_token()

        # Calculate date range using configured time frame
        today = datetime.now(timezone.utc)
        cutoff_date = today - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

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

                # Include if within configured time frame
                if cutoff_date <= file_date <= today:
                    recent_files.append(filename)
            except Exception:
                continue

        return recent_files

    except Exception as e:
        print(f"Error getting daily files: {str(e)}")
        return []


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

                    # Extract github_identifier from filename (e.g., "agent[bot].json" -> "agent[bot]")
                    filename_identifier = json_file.replace('.json', '')

                    # Add or override github_identifier to match filename
                    agent_data['github_identifier'] = filename_identifier

                    # Normalize name field: use 'name' if exists, otherwise use identifier
                    if 'name' in agent_data:
                        agent_data['agent_name'] = agent_data['name']
                    elif 'agent_name' not in agent_data:
                        agent_data['agent_name'] = filename_identifier

                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
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




# =============================================================================
# DATA MANAGEMENT
# =============================================================================

def mine_all_agents():
    """
    Mine issue metadata for all agents within UPDATE_TIME_FRAME_DAYS and save to HuggingFace.
    Uses ONE BigQuery query for ALL agents (most efficient approach).

    Runs periodically based on UPDATE_TIME_FRAME_DAYS (e.g., weekly).
    """
    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return

    # Extract all identifiers
    identifiers = [agent['github_identifier'] for agent in agents if agent.get('github_identifier')]
    if not identifiers:
        print("No valid agent identifiers found")
        return

    print(f"\n{'='*80}")
    print(f"⛏️  [MINE] Starting BigQuery data mining for {len(identifiers)} agents")
    print(f"Time frame: Last {LEADERBOARD_TIME_FRAME_DAYS} days")
    print(f"Data source: BigQuery + GitHub Archive (ONE QUERY FOR ALL AGENTS)")
    print(f"⚠️  This will query BigQuery and may take several minutes")
    print(f"{'='*80}\n")

    # Initialize BigQuery client
    try:
        client = get_bigquery_client()
    except Exception as e:
        print(f"✗ Failed to initialize BigQuery client: {str(e)}")
        return

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    try:
        all_metadata = fetch_all_issue_metadata_single_query(
            client, identifiers, start_date, end_date
        )
    except Exception as e:
        print(f"✗ Error during BigQuery fetch: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Save results for each agent
    print(f"\n{'='*80}")
    print(f"💾 Saving results to HuggingFace for each agent...")
    print(f"{'='*80}\n")

    success_count = 0
    error_count = 0
    no_data_count = 0

    for i, agent in enumerate(agents, 1):
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        if not identifier:
            print(f"[{i}/{len(agents)}] Skipping agent without identifier")
            error_count += 1
            continue

        metadata = all_metadata.get(identifier, [])

        print(f"[{i}/{len(agents)}] {agent_name} ({identifier}):")

        try:
            if metadata:
                print(f"   💾 Saving {len(metadata)} issue records...")
                if save_issue_metadata_to_hf(metadata, identifier):
                    success_count += 1
                else:
                    error_count += 1
            else:
                print(f"   No issues found")
                no_data_count += 1

        except Exception as e:
            print(f"   ✗ Error saving {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    print(f"\n{'='*80}")
    print(f"✅ Mining complete!")
    print(f"   Total agents: {len(agents)}")
    print(f"   Successfully saved: {success_count}")
    print(f"   No data (skipped): {no_data_count}")
    print(f"   Errors: {error_count}")
    print(f"   BigQuery queries executed: 1")
    print(f"{'='*80}\n")


def construct_leaderboard_from_metadata():
    """
    Construct leaderboard from stored issue metadata instead of fetching all issues.
    Much more memory-efficient and faster.

    Returns dictionary of agent stats.
    """
    print("📊 Constructing leaderboard from issue metadata...")
    # Load agents
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found")
        return {}

    # Load all issue metadata
    all_metadata = load_issue_metadata()

    cache_dict = {}

    for agent in agents:
        identifier = agent.get('github_identifier')
        agent_name = agent.get('agent_name', 'Unknown')

        # Filter metadata for this agent
        agent_metadata = [issue for issue in all_metadata if issue.get('agent_identifier') == identifier]

        # Calculate stats
        stats = calculate_issue_stats_from_metadata(agent_metadata)

        cache_dict[identifier] = {
            'agent_name': agent_name,
            'website': agent.get('website', 'N/A'),
            'github_identifier': identifier,
            **stats
        }

    return cache_dict


# =============================================================================
# UI FUNCTIONS
# =============================================================================

def generate_color(index, total):
    """Generate distinct colors using HSL color space for better distribution"""
    hue = (index * 360 / total) % 360
    saturation = 70 + (index % 3) * 10  # Vary saturation slightly
    lightness = 45 + (index % 2) * 10   # Vary lightness slightly
    return f'hsl({hue}, {saturation}%, {lightness}%)'


def create_monthly_metrics_plot():
    """
    Create a Plotly figure with dual y-axes showing:
    - Left y-axis: Resolved Rate (%) as line curves
    - Right y-axis: Total Issues created as bar charts

    Each agent gets a unique color for both their line and bars.
    Shows only top 5 agents by total issue count.
    """
    metrics = calculate_monthly_metrics_by_agent(top_n=5)

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

    agents = metrics['agents']
    months = metrics['months']
    data = metrics['data']

    # Generate unique colors for many agents using HSL color space
    agent_colors = {agent: generate_color(idx, len(agents)) for idx, agent in enumerate(agents)}

    # Add traces for each agent
    for agent_name in agents:
        color = agent_colors[agent_name]
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
                    name=agent_name,
                    marker=dict(color=color, opacity=0.6),
                    legendgroup=agent_name,
                    showlegend=False,  # Don't show in legend (already shown for line)
                    hovertemplate='<b>%{fullData.name}</b><br>' +
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
        hovermode='closest',
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
    Construct leaderboard from issue metadata and convert to pandas DataFrame for display.
    Returns formatted DataFrame sorted by total issues.
    """
    # Construct leaderboard from metadata
    cache_dict = construct_leaderboard_from_metadata()

    if not cache_dict:
        # Return empty DataFrame with correct columns if no data
        column_names = [col[0] for col in LEADERBOARD_COLUMNS]
        return pd.DataFrame(columns=column_names)

    rows = []
    for data in cache_dict.values():
        # Filter out agents with zero total issues
        if data.get('total_issues', 0) == 0:
            continue
        # Only include display-relevant fields
        rows.append([
            data.get('agent_name', 'Unknown'),
            data.get('website', 'N/A'),
            data.get('total_issues', 0),
            data.get('resolved_issues', 0),
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

    # Sort by Total Issues descending
    if "Total Issues" in df.columns and not df.empty:
        df = df.sort_values(by="Total Issues", ascending=False).reset_index(drop=True)

    return df


def submit_agent(identifier, agent_name, developer, website):
    """
    Submit a new agent to the leaderboard.
    Validates input and saves submission. Issue data will be populated by daily incremental updates.
    """
    # Validate required fields
    if not identifier or not identifier.strip():
        return "❌ GitHub identifier is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not agent_name or not agent_name.strip():
        return "❌ Agent name is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not developer or not developer.strip():
        return "❌ Developer name is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()
    if not website or not website.strip():
        return "❌ Website URL is required", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    # Clean inputs
    identifier = identifier.strip()
    agent_name = agent_name.strip()
    developer = developer.strip()
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
        'developer': developer,
        'github_identifier': identifier,
        'website': website,
    }

    # Save to HuggingFace
    if not save_agent_to_hf(submission):
        return "❌ Failed to save submission", get_leaderboard_dataframe(), create_monthly_metrics_plot()

    return f"✅ Successfully submitted {agent_name}! Issue data will be populated by daily incremental updates.", get_leaderboard_dataframe(), create_monthly_metrics_plot()


# =============================================================================
# BACKGROUND TASKS
# =============================================================================


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

# Start APScheduler for periodic issue mining via BigQuery
# NOTE: On app startup, we only LOAD existing cached data from HuggingFace
# Mining (BigQuery queries) ONLY happens on schedule (weekly on Mondays)
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    mine_all_agents,
    trigger=CronTrigger(day_of_week='mon', hour=0, minute=0),  # Every Monday at 12:00 AM UTC
    id='periodic_bigquery_mining',
    name='Periodic BigQuery Issue Mining',
    replace_existing=True
)
scheduler.start()
print(f"\n{'='*80}")
print(f"✓ Scheduler initialized successfully")
print(f"⛏️  Mining schedule: Every Monday at 12:00 AM UTC")
print(f"📥 On startup: Only loads cached data from HuggingFace (no mining)")
print(f"{'='*80}\n")

# Create Gradio interface
with gr.Blocks(title="SWE Agent Issue Leaderboard", theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🏆 SWE Agent Issue Leaderboard")
    gr.Markdown(f"Track and compare GitHub issue resolution statistics for SWE agents")

    with gr.Tabs():

        # Leaderboard Tab
        with gr.Tab("📊 Leaderboard"):
            gr.Markdown(f"*All statistics are based on issues from the last {LEADERBOARD_TIME_FRAME_DAYS // 30} months*")
            leaderboard_table = Leaderboard(
                value=get_leaderboard_dataframe(),
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
                filter_columns=[
                    ColumnFilter(
                        "Acceptance Rate (%)",
                        min=0,
                        max=100,
                        default=[0, 100],
                        type="slider",
                        label="Acceptance Rate (%)"
                    )
                ]
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
                    developer_input = gr.Textbox(
                        label="Developer*",
                        placeholder="Your developer or team name"
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
                inputs=[github_input, name_input, developer_input, website_input],
                outputs=[submission_status, leaderboard_table, monthly_plot]
            )


# Launch application
if __name__ == "__main__":
    app.launch()