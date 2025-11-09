"""
Minimalist Issue Metadata Mining Script
Mines issue metadata from GitHub Archive via BigQuery and saves to HuggingFace dataset.
"""

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from dotenv import load_dotenv
from google.cloud import bigquery
import backoff

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/bot_metadata"
ISSUE_METADATA_REPO = "SWE-Arena/issue_metadata"
LEADERBOARD_REPO = "SWE-Arena/leaderboard_metadata"
LEADERBOARD_TIME_FRAME_DAYS = 180  # Time frame for leaderboard

# =============================================================================
# HUGGINGFACE API WRAPPERS WITH BACKOFF
# =============================================================================

def is_rate_limit_error(e):
    """Check if the exception is a rate limit error (429)."""
    return isinstance(e, HfHubHTTPError) and e.response.status_code == 429

@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_tries=8,
    base=300,
    max_value=3600,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"   ⏳ Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/{8}...")
)
def upload_large_folder_with_backoff(api, **kwargs):
    """Upload large folder with exponential backoff on rate limit errors."""
    return api.upload_large_folder(**kwargs)

@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_tries=8,
    base=300,
    max_value=3600,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"   ⏳ Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/{8}...")
)
def list_repo_files_with_backoff(api, **kwargs):
    """List repo files with exponential backoff on rate limit errors."""
    return api.list_repo_files(**kwargs)

@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_tries=8,
    base=300,
    max_value=3600,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"   ⏳ Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/{8}...")
)
def hf_hub_download_with_backoff(**kwargs):
    """Download from HF Hub with exponential backoff on rate limit errors."""
    return hf_hub_download(**kwargs)

@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_tries=8,
    base=300,
    max_value=3600,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"   ⏳ Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/{8}...")
)
def upload_file_with_backoff(api, **kwargs):
    """Upload file with exponential backoff on rate limit errors."""
    return api.upload_file(**kwargs)

@backoff.on_exception(
    backoff.expo,
    HfHubHTTPError,
    giveup=lambda e: not is_rate_limit_error(e),
    max_tries=8,
    base=300,
    max_value=3600,
    jitter=backoff.full_jitter,
    on_backoff=lambda details: print(f"   ⏳ Rate limited. Retrying in {details['wait']/60:.1f} minutes ({details['wait']:.0f}s) - attempt {details['tries']}/{8}...")
)
def upload_folder_with_backoff(api, **kwargs):
    """Upload folder with exponential backoff on rate limit errors."""
    return api.upload_folder(**kwargs)

# =============================================================================
# UTILITY FUNCTIONS
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
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
    return data


def save_jsonl(filename, data):
    """Save list of dictionaries to JSONL file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


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
    Generate UNION ALL statements for githubarchive.month tables in date range.

    Args:
        start_date: Start datetime
        end_date: End datetime

    Returns:
        String with UNION ALL SELECT statements for all monthly tables in range
    """
    table_names = []

    # Start from the beginning of start_date's month
    current_date = start_date.replace(day=1)
    end_month = end_date.replace(day=1)

    while current_date <= end_month:
        table_name = f"`githubarchive.month.{current_date.strftime('%Y%m')}`"
        table_names.append(table_name)

        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    # Create UNION ALL chain
    union_parts = [f"SELECT * FROM {table}" for table in table_names]
    return " UNION ALL ".join(union_parts)


# =============================================================================
# BIGQUERY FUNCTIONS
# =============================================================================

def fetch_issue_metadata_batched(client, identifiers, start_date, end_date, batch_size=100, upload_immediately=True):
    """
    Fetch issue metadata for ALL agents using BATCHED BigQuery queries.

    Splits agents into smaller batches to avoid performance issues with large UNNEST arrays
    and correlated subqueries. Each batch query runs much faster than one massive query.

    Args:
        client: BigQuery client instance
        identifiers: List of GitHub usernames/bot identifiers
        start_date: Start datetime (timezone-aware)
        end_date: End datetime (timezone-aware)
        batch_size: Number of agents per batch (default: 100)
        upload_immediately: Upload results to HuggingFace immediately after each batch (default: True)

    Returns:
        Dictionary mapping agent identifier to list of issue metadata
    """
    print(f"\n🔍 Querying BigQuery for {len(identifiers)} agents using BATCHED approach")
    print(f"   Batch size: {batch_size} agents per query")
    print(f"   Upload mode: {'Immediate (after each batch)' if upload_immediately else 'Deferred (after all batches)'}")

    # Split identifiers into batches
    batches = [identifiers[i:i + batch_size] for i in range(0, len(identifiers), batch_size)]
    print(f"   Total batches: {len(batches)}")

    # Collect results from all batches
    all_metadata = {}

    for batch_num, batch_identifiers in enumerate(batches, 1):
        print(f"\n{'─'*80}")
        print(f"📦 Processing Batch {batch_num}/{len(batches)} ({len(batch_identifiers)} agents)")
        print(f"{'─'*80}")

        try:
            batch_results = fetch_all_issue_metadata_single_query(
                client, batch_identifiers, start_date, end_date
            )

            # Merge results
            for identifier, metadata_list in batch_results.items():
                if identifier in all_metadata:
                    all_metadata[identifier].extend(metadata_list)
                else:
                    all_metadata[identifier] = metadata_list

            print(f"   ✓ Batch {batch_num} completed: {len(batch_results)} agents with data")

            # Upload immediately after this batch if enabled
            if upload_immediately and batch_results:
                print(f"\n   🤗 Uploading batch {batch_num}/{len(batches)} results to HuggingFace...")
                upload_success = 0
                upload_errors = 0

                for identifier, metadata_list in batch_results.items():
                    if metadata_list:
                        if save_issue_metadata_to_hf(metadata_list, identifier):
                            upload_success += 1
                        else:
                            upload_errors += 1

                print(f"   ✓ Batch {batch_num}/{len(batches)} upload complete ({upload_success} agents uploaded, {upload_errors} errors)")

        except Exception as e:
            print(f"   ✗ Batch {batch_num} failed: {str(e)}")
            print(f"   Continuing with remaining batches...")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ All batches completed!")
    print(f"   Total agents with data: {len(all_metadata)}")
    total_issues = sum(len(issues) for issues in all_metadata.values())
    print(f"   Total issues found: {total_issues}")
    print(f"{'='*80}\n")

    return all_metadata


def fetch_all_issue_metadata_single_query(client, identifiers, start_date, end_date):
    """
    Fetch issue metadata for a batch of agents using ONE comprehensive BigQuery query.

    This query fetches IssuesEvent and IssueCommentEvent from GitHub Archive and
    deduplicates to get the latest state of each issue. Filters by issue author,
    commenter, or assignee.

    NOTE: This function is designed for smaller batches (~100 agents). For large
    numbers of agents, use fetch_issue_metadata_batched() instead.

    Args:
        client: BigQuery client instance
        identifiers: List of GitHub usernames/bot identifiers (recommended: <100)
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
    print(f"\n🔍 Querying BigQuery for {len(identifiers)} agents in SINGLE QUERY")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Generate table UNION statements for issue events
    issue_tables = generate_table_union_statements(start_date, end_date)

    # Build identifier list (handle both bot and non-bot versions)
    identifier_set = set()
    for id in identifiers:
        identifier_set.add(id)
        # Also add stripped version without [bot] suffix
        stripped = id.replace('[bot]', '')
        if stripped != id:
            identifier_set.add(stripped)

    # Convert to array literal for UNNEST (avoids query size limits from large IN clauses)
    identifier_array = '[' + ', '.join([f'"{id}"' for id in identifier_set]) + ']'

    print(f"   Total identifiers (including bot/non-bot variants): {len(identifier_set)}")

    # Build comprehensive query with CTEs using UNNEST instead of large IN clauses
    query = f"""
    WITH agent_identifiers AS (
      -- Create a table from the identifier array to avoid massive IN clauses
      SELECT identifier
      FROM UNNEST({identifier_array}) AS identifier
    ),

    issue_events AS (
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
        -- Filter by author OR commenter OR assignee using JOIN instead of IN
        AND (
          JSON_EXTRACT_SCALAR(payload, '$.issue.user.login') IN (SELECT identifier FROM agent_identifiers)
          OR JSON_EXTRACT_SCALAR(payload, '$.comment.user.login') IN (SELECT identifier FROM agent_identifiers)
          OR JSON_EXTRACT_SCALAR(payload, '$.issue.assignee.login') IN (SELECT identifier FROM agent_identifiers)
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
          WHEN author IN (SELECT identifier FROM agent_identifiers) THEN author
          WHEN commenter IN (SELECT identifier FROM agent_identifiers) THEN commenter
          WHEN assignee IN (SELECT identifier FROM agent_identifiers) THEN assignee
          ELSE NULL
        END as agent_identifier,
        url,
        created_at,
        closed_at,
        state_reason
      FROM latest_states
      WHERE
        author IN (SELECT identifier FROM agent_identifiers)
        OR commenter IN (SELECT identifier FROM agent_identifiers)
        OR assignee IN (SELECT identifier FROM agent_identifiers)
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
# HUGGINGFACE STORAGE FUNCTIONS
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

    This function OVERWRITES existing files completely with fresh data from BigQuery.
    Uses upload_folder for single-commit batch uploads (avoids rate limit issues).

    Args:
        metadata_list: List of issue metadata dictionaries
        agent_identifier: GitHub identifier of the agent (used as folder name)
    """
    import shutil

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)

        # Group by date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        if not grouped:
            print(f"   No valid metadata to save for {agent_identifier}")
            return False

        # Create a temporary directory for batch upload
        temp_dir = tempfile.mkdtemp()
        agent_folder = os.path.join(temp_dir, agent_identifier)
        os.makedirs(agent_folder, exist_ok=True)

        try:
            print(f"   📦 Preparing batch upload for {len(grouped)} daily files...")

            # Process each daily file
            for (issue_year, month, day), day_metadata in grouped.items():
                filename = f"{agent_identifier}/{issue_year}.{month:02d}.{day:02d}.jsonl"
                local_filename = os.path.join(agent_folder, f"{issue_year}.{month:02d}.{day:02d}.jsonl")

                # Sort by created_at for better organization
                day_metadata.sort(key=lambda x: x.get('created_at', ''), reverse=True)

                # Save to temp directory (complete overwrite, no merging)
                save_jsonl(local_filename, day_metadata)
                print(f"      Prepared {len(day_metadata)} issues for {filename}")

            # Upload entire folder using upload_folder (single commit per agent)
            print(f"   🤗 Uploading {len(grouped)} files ({len(metadata_list)} total issues)...")
            upload_folder_with_backoff(
                api,
                folder_path=temp_dir,
                repo_id=ISSUE_METADATA_REPO,
                repo_type="dataset",
                commit_message=f"Update issue metadata for {agent_identifier} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )
            print(f"   ✓ Batch upload complete for {agent_identifier}")

            return True

        finally:
            # Always clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"   ✗ Error saving issue metadata: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def load_agents_from_hf():
    """
    Load all agent metadata JSON files from HuggingFace dataset.

    The github_identifier is extracted from the filename (e.g., 'agent-name[bot].json' -> 'agent-name[bot]')
    """
    try:
        api = HfApi()
        agents = []

        # List all files in the repository
        files = list_repo_files_with_backoff(api, repo_id=AGENTS_REPO, repo_type="dataset")

        # Filter for JSON files only
        json_files = [f for f in files if f.endswith('.json')]

        print(f"Found {len(json_files)} agent files in {AGENTS_REPO}")

        # Download and parse each JSON file
        for json_file in json_files:
            try:
                file_path = hf_hub_download_with_backoff(
                    repo_id=AGENTS_REPO,
                    filename=json_file,
                    repo_type="dataset"
                )

                with open(file_path, 'r') as f:
                    agent_data = json.load(f)

                    # Only process agents with status == "public"
                    if agent_data.get('status') != 'public':
                        continue

                    # Extract github_identifier from filename (remove .json extension)
                    github_identifier = json_file.replace('.json', '')
                    agent_data['github_identifier'] = github_identifier

                    agents.append(agent_data)

            except Exception as e:
                print(f"Warning: Could not load {json_file}: {str(e)}")
                continue

        print(f"✓ Loaded {len(agents)} agents from HuggingFace")
        return agents

    except Exception as e:
        print(f"Could not load agents from HuggingFace: {str(e)}")
        return []


# =============================================================================
# LEADERBOARD CALCULATION FUNCTIONS
# =============================================================================

def calculate_issue_stats_from_metadata(metadata_list):
    """
    Calculate statistics from a list of issue metadata.

    Returns:
        dict: Issue statistics including total, closed, resolved counts and rate
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


def calculate_monthly_metrics(all_metadata, agents):
    """
    Calculate monthly metrics for all agents for visualization.

    Args:
        all_metadata: Dictionary mapping agent_identifier to list of issue metadata
        agents: List of agent dictionaries with metadata

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
    # Create mapping from agent_identifier to agent_name
    identifier_to_name = {
        agent.get('github_identifier'): agent.get('name', agent.get('name', agent.get('github_identifier')))
        for agent in agents if agent.get('github_identifier')
    }

    # Group by agent and month
    agent_month_data = defaultdict(lambda: defaultdict(list))

    for identifier, metadata_list in all_metadata.items():
        agent_name = identifier_to_name.get(identifier, identifier)

        for issue_meta in metadata_list:
            created_at = issue_meta.get('created_at')
            if not created_at:
                continue

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

    agents_list = sorted(list(agent_month_data.keys()))

    return {
        'agents': agents_list,
        'months': months,
        'data': result_data
    }


def save_leaderboard_and_metrics_to_hf(all_metadata, agents):
    """
    Creates a comprehensive JSON file with both leaderboard stats and monthly metrics.
    If the file exists, it will be overwritten.

    Args:
        all_metadata: Dictionary mapping agent_identifier to list of issue metadata
        agents: List of agent dictionaries with metadata

    Returns:
        bool: True if successful, False otherwise
    """
    import io

    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi(token=token)

        print(f"\n{'='*80}")
        print(f"📊 Preparing leaderboard and metrics data for upload...")
        print(f"{'='*80}\n")

        # Build leaderboard data
        print("   Constructing leaderboard data...")
        leaderboard_data = {}

        for agent in agents:
            identifier = agent.get('github_identifier')
            agent_name = agent.get('name', 'Unknown')

            if not identifier:
                continue

            metadata = all_metadata.get(identifier, [])
            stats = calculate_issue_stats_from_metadata(metadata)

            leaderboard_data[identifier] = {
                'name': agent_name,
                'website': agent.get('website', 'N/A'),
                'github_identifier': identifier,
                **stats
            }

        # Get monthly metrics data
        print("   Calculating monthly metrics...")
        monthly_metrics = calculate_monthly_metrics(all_metadata, agents)

        # Combine into a single structure
        combined_data = {
            "leaderboard": leaderboard_data,
            "monthly_metrics": monthly_metrics,
            "metadata": {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "time_frame_days": LEADERBOARD_TIME_FRAME_DAYS,
                "total_agents": len(leaderboard_data)
            }
        }

        print(f"   Leaderboard entries: {len(leaderboard_data)}")
        print(f"   Monthly metrics for: {len(monthly_metrics['agents'])} agents")
        print(f"   Time frame: {LEADERBOARD_TIME_FRAME_DAYS} days")

        # Convert to JSON and create file-like object
        json_content = json.dumps(combined_data, indent=2)
        file_like_object = io.BytesIO(json_content.encode('utf-8'))

        # Upload to HuggingFace (will overwrite if exists)
        print(f"\n🤗 Uploading to {LEADERBOARD_REPO}...")
        upload_file_with_backoff(
            api,
            path_or_fileobj=file_like_object,
            path_in_repo="swe-issue.json",
            repo_id=LEADERBOARD_REPO,
            repo_type="dataset",
            token=token,
            commit_message=f"Update leaderboard data - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )

        print(f"   ✓ Successfully uploaded swe-issue.json")
        print(f"{'='*80}\n")

        return True

    except Exception as e:
        print(f"✗ Error saving leaderboard and metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# MAIN MINING FUNCTION
# =============================================================================

def mine_all_agents():
    """
    Mine issue metadata for all agents within LEADERBOARD_TIME_FRAME_DAYS and save to HuggingFace.
    Uses ONE BigQuery query for ALL agents (most efficient approach).
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
    print(f"Starting issue metadata mining for {len(identifiers)} agents")
    print(f"Time frame: Last {LEADERBOARD_TIME_FRAME_DAYS} days")
    print(f"Data source: BigQuery + GitHub Archive (BATCHED QUERIES)")
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
        # Use batched approach for better performance
        # upload_immediately=True means each batch uploads to HuggingFace right after BigQuery completes
        all_metadata = fetch_issue_metadata_batched(
            client, identifiers, start_date, end_date, batch_size=100, upload_immediately=True
        )

        # Calculate summary statistics
        total_prs = sum(len(metadata_list) for metadata_list in all_metadata.values())
        agents_with_data = sum(1 for metadata_list in all_metadata.values() if metadata_list)

        print(f"\n{'='*80}")
        print(f"✅ BigQuery mining and upload complete!")
        print(f"   Total agents: {len(agents)}")
        print(f"   Agents with data: {agents_with_data}")
        print(f"   Total PRs found: {total_prs}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"✗ Error during BigQuery fetch: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # After mining is complete, save leaderboard and metrics to HuggingFace
    print(f"📤 Uploading leaderboard and metrics data...")
    if save_leaderboard_and_metrics_to_hf(all_metadata, agents):
        print(f"✓ Leaderboard and metrics successfully uploaded to {LEADERBOARD_REPO}")
    else:
        print(f"⚠️ Failed to upload leaderboard and metrics data")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()
