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
from dotenv import load_dotenv
from google.cloud import bigquery

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/swe_agents"
ISSUE_METADATA_REPO = "SWE-Arena/issue_metadata"
LEADERBOARD_TIME_FRAME_DAYS = 3  # Time frame for leaderboard

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


# =============================================================================
# BIGQUERY FUNCTIONS
# =============================================================================

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
    Uses batch upload to avoid rate limit (uploads entire folder in single commit).

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

            # Upload entire folder using upload_large_folder (optimized for large files)
            # Note: upload_large_folder creates multiple commits automatically and doesn't support custom commit_message
            print(f"   🤗 Uploading {len(grouped)} files ({len(metadata_list)} total issues)...")
            api.upload_large_folder(
                folder_path=temp_dir,
                repo_id=ISSUE_METADATA_REPO,
                repo_type="dataset"
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

                    # Only process agents with status == "public"
                    if agent_data.get('status') != 'public':
                        print(f"Skipping {json_file}: status is not 'public'")
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
    print(f"Data source: BigQuery + GitHub Archive (ONE QUERY FOR ALL AGENTS)")
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
        agent_name = agent.get('name', agent.get('agent_name', 'Unknown'))

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


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()
