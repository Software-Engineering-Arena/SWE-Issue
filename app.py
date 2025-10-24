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
DEBUG_ISSUE_METADATA_CACHE = defaultdict(list)

AGENTS_REPO = "SWE-Arena/swe_agents"  # HuggingFace dataset for agent metadata
ISSUE_METADATA_REPO = "SWE-Arena/issue_metadata"  # HuggingFace dataset for issue metadata
LEADERBOARD_TIME_FRAME_DAYS = 180  # Time frame for leaderboard (past 6 months)

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

def request_with_backoff(method, url, *, headers=None, params=None, json_body=None, data=None, max_retries=10, timeout=30, token_pool=None, token=None):
    """
    Perform an HTTP request with exponential backoff and jitter for GitHub API.
    Retries on 403/429 (rate limits), 5xx server errors, and transient network exceptions.

    Args:
        token_pool: Optional TokenPool instance for automatic rate limit tracking
        token: Optional token being used (for marking as rate-limited)

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
                reset_timestamp = None

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
                            reset_timestamp = reset_ts
                            wait = max(reset_ts - time.time() + 2, 1)
                        except Exception:
                            wait = None

                # Mark token as rate-limited if we have token_pool and token
                if status in (403, 429) and token_pool and token:
                    token_pool.mark_rate_limited(token, reset_timestamp)

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

def get_github_tokens():
    """Get all GitHub tokens from environment variables (all keys starting with GITHUB_TOKEN)."""
    tokens = []
    for key, value in os.environ.items():
        if key.startswith('GITHUB_TOKEN') and value:
            tokens.append(value)

    if not tokens:
        print("Warning: No GITHUB_TOKEN found. API rate limits: 60/hour (authenticated: 5000/hour)")
    else:
        print(f"✓ Loaded {len(tokens)} GitHub token(s) for rotation")

    return tokens


def get_github_token():
    """Get primary GitHub token from environment variables (backward compatibility)."""
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Warning: GITHUB_TOKEN not found. API rate limits: 60/hour (authenticated: 5000/hour)")
    return token


class TokenPool:
    """
    Hybrid token pool with parallel execution and round-robin fallback.

    Splits tokens into two pools:
    - 50% for parallel execution (maximize throughput)
    - 50% for round-robin backup (handle rate limits)

    Features:
    - Automatic rate limit detection and tracking
    - Token recovery when rate limits expire
    - Statistics monitoring
    - Thread-safe operations
    """
    def __init__(self, tokens):
        import threading

        # Store all tokens
        self.all_tokens = tokens if tokens else [None]
        total_tokens = len(self.all_tokens)

        # Split tokens into parallel and round-robin pools (50/50)
        # For odd numbers, round-robin gets the extra token
        split_point = max(1, total_tokens // 2)

        self.parallel_tokens = self.all_tokens[:split_point]
        self.roundrobin_tokens = self.all_tokens[split_point:] if split_point < total_tokens else self.all_tokens

        # Round-robin index for fallback pool
        self.roundrobin_index = 0

        # Track rate-limited tokens with reset timestamps
        self.rate_limited_tokens = {}  # {token: reset_timestamp}

        # Statistics
        self.stats = {
            'parallel_calls': 0,
            'roundrobin_calls': 0,
            'fallback_triggers': 0
        }

        # Thread lock for thread-safety
        self.lock = threading.Lock()

        print(f"🔀 Token Pool Initialized:")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Parallel pool: {len(self.parallel_tokens)} tokens")
        print(f"   Round-robin pool: {len(self.roundrobin_tokens)} tokens")

    def _clean_expired_rate_limits(self):
        """Remove tokens from rate-limited set if their reset time has passed."""
        current_time = time.time()
        expired = [token for token, reset_time in self.rate_limited_tokens.items()
                  if reset_time and current_time >= reset_time]
        for token in expired:
            del self.rate_limited_tokens[token]
            print(f"   ✓ Token recovered from rate limit")

    def get_parallel_token(self):
        """Get an available token from the parallel pool."""
        with self.lock:
            self._clean_expired_rate_limits()

            # Find first available parallel token (not rate-limited)
            for token in self.parallel_tokens:
                if token not in self.rate_limited_tokens:
                    self.stats['parallel_calls'] += 1
                    return token

            # All parallel tokens are rate-limited
            return None

    def get_roundrobin_token(self):
        """Get next token from round-robin pool."""
        with self.lock:
            self._clean_expired_rate_limits()

            if not self.roundrobin_tokens:
                return None

            # Try up to pool size to find non-rate-limited token
            attempts = 0
            max_attempts = len(self.roundrobin_tokens)

            while attempts < max_attempts:
                token = self.roundrobin_tokens[self.roundrobin_index]
                self.roundrobin_index = (self.roundrobin_index + 1) % len(self.roundrobin_tokens)
                attempts += 1

                if token not in self.rate_limited_tokens:
                    self.stats['roundrobin_calls'] += 1
                    return token

            # All round-robin tokens are rate-limited, return one anyway
            # (request_with_backoff will handle the rate limit)
            token = self.roundrobin_tokens[self.roundrobin_index]
            self.roundrobin_index = (self.roundrobin_index + 1) % len(self.roundrobin_tokens)
            self.stats['roundrobin_calls'] += 1
            return token

    def get_next_token(self):
        """
        Get next available token using hybrid strategy:
        1. Try parallel pool first
        2. Fall back to round-robin if parallel is exhausted
        """
        # Try parallel pool first
        token = self.get_parallel_token()

        if token is not None:
            return token

        # Parallel pool exhausted, fall back to round-robin
        with self.lock:
            self.stats['fallback_triggers'] += 1

        return self.get_roundrobin_token()

    def get_headers(self):
        """Get headers with the next token in rotation."""
        token = self.get_next_token()
        return {'Authorization': f'token {token}'} if token else {}

    def mark_rate_limited(self, token, reset_timestamp=None):
        """
        Mark a token as rate-limited with optional reset timestamp.

        Args:
            token: The token to mark
            reset_timestamp: Unix timestamp when rate limit resets (optional)
        """
        with self.lock:
            self.rate_limited_tokens[token] = reset_timestamp
            pool_type = "parallel" if token in self.parallel_tokens else "round-robin"
            if reset_timestamp:
                reset_time = datetime.fromtimestamp(reset_timestamp, timezone.utc).strftime('%H:%M:%S UTC')
                print(f"   ⚠️ Token marked as rate-limited ({pool_type} pool, resets at {reset_time})")
            else:
                print(f"   ⚠️ Token marked as rate-limited ({pool_type} pool)")

    def get_available_parallel_tokens(self):
        """Get list of all available (non-rate-limited) parallel tokens."""
        with self.lock:
            self._clean_expired_rate_limits()
            return [token for token in self.parallel_tokens
                   if token not in self.rate_limited_tokens]

    def get_stats(self):
        """Get current statistics."""
        with self.lock:
            self._clean_expired_rate_limits()
            parallel_rate_limited = sum(1 for t in self.parallel_tokens
                                       if t in self.rate_limited_tokens)
            roundrobin_rate_limited = sum(1 for t in self.roundrobin_tokens
                                         if t in self.rate_limited_tokens)

            return {
                **self.stats,
                'parallel_rate_limited': parallel_rate_limited,
                'roundrobin_rate_limited': roundrobin_rate_limited
            }

    def print_stats(self):
        """Print statistics about token pool usage."""
        stats = self.get_stats()
        total_calls = stats['parallel_calls'] + stats['roundrobin_calls']

        if total_calls == 0:
            print("📊 No API calls made yet")
            return

        parallel_pct = (stats['parallel_calls'] / total_calls * 100) if total_calls > 0 else 0
        roundrobin_pct = (stats['roundrobin_calls'] / total_calls * 100) if total_calls > 0 else 0

        print(f"📊 Token Pool Statistics:")
        print(f"   Total API calls: {total_calls}")
        print(f"   Parallel calls: {stats['parallel_calls']} ({parallel_pct:.1f}%)")
        print(f"   Round-robin calls: {stats['roundrobin_calls']} ({roundrobin_pct:.1f}%)")
        print(f"   Fallback triggers: {stats['fallback_triggers']}")
        print(f"   Currently rate-limited: {stats['parallel_rate_limited']} parallel, {stats['roundrobin_rate_limited']} round-robin")


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


def fetch_issues_parallel(query_patterns, start_date, end_date, token_pool, issues_by_id, debug_limit=None):
    """
    Fetch issues for multiple query patterns in parallel using available parallel tokens.

    Args:
        query_patterns: List of query patterns to search
        start_date: Start date for time range
        end_date: End date for time range
        token_pool: TokenPool instance for token management
        issues_by_id: Shared dictionary to store issues (thread-safe operations)
        debug_limit: If set, stops fetching after this many issues per pattern

    Returns:
        Total number of issues found across all patterns
    """
    import concurrent.futures
    import threading

    # Get available parallel tokens
    available_tokens = token_pool.get_available_parallel_tokens()

    if not available_tokens:
        print("   ⚠️ No parallel tokens available, using sequential fallback")
        total_found = 0
        for pattern in query_patterns:
            count = fetch_issues_with_time_partition(
                pattern, start_date, end_date, token_pool, issues_by_id, debug_limit, depth=0
            )
            total_found += count
        return total_found

    # Determine max workers based on available tokens
    max_workers = min(len(query_patterns), len(available_tokens))

    print(f"   🚀 Using parallel execution with {max_workers} workers")

    # Thread-safe lock for issues_by_id updates
    lock = threading.Lock()

    def fetch_pattern(pattern, token):
        """Worker function to fetch issues for a single pattern."""
        # Create temporary dict for this pattern
        pattern_issues = {}

        try:
            # Fetch issues for this pattern
            count = fetch_issues_with_time_partition(
                pattern,
                start_date,
                end_date,
                token_pool,
                pattern_issues,
                debug_limit,
                depth=0
            )

            # Merge into shared dict with lock
            with lock:
                for issue_id, issue in pattern_issues.items():
                    if issue_id not in issues_by_id:
                        issues_by_id[issue_id] = issue

            return count

        except Exception as e:
            print(f"   ✗ Error in parallel fetch for pattern '{pattern}': {str(e)}")
            return 0

    # Execute patterns in parallel
    total_found = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map patterns to tokens
        futures = []
        for i, pattern in enumerate(query_patterns):
            token = available_tokens[i % len(available_tokens)]
            future = executor.submit(fetch_pattern, pattern, token)
            futures.append(future)

        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                count = future.result()
                total_found += count
            except Exception as e:
                print(f"   ✗ Parallel execution error: {str(e)}")

    return total_found


def fetch_issues_with_time_partition(base_query, start_date, end_date, token_pool, issues_by_id, debug_limit=None, depth=0):
    """
    Fetch issues within a specific time range using time-based partitioning.
    Recursively splits the time range if hitting the 1000-result limit.
    Supports splitting by day, hour, minute, and second as needed.

    Args:
        base_query: Base GitHub search query
        start_date: Start date for time range
        end_date: End date for time range
        token_pool: TokenPool instance for rotating tokens
        issues_by_id: Dictionary to store issues (deduplicated by ID)
        debug_limit: If set, stops fetching after this many issues (for testing)
        depth: Current recursion depth (for tracking)

    Returns the number of issues found in this time partition.
    """
    # Calculate time difference
    time_diff = end_date - start_date
    total_seconds = time_diff.total_seconds()

    # Determine granularity and format dates accordingly
    if total_seconds >= 86400:  # >= 1 day
        # Use day granularity (YYYY-MM-DD)
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
    elif total_seconds >= 3600:  # >= 1 hour but < 1 day
        # Use hour granularity (YYYY-MM-DDTHH:MM:SSZ)
        start_str = start_date.strftime('%Y-%m-%dT%H:00:00Z')
        end_str = end_date.strftime('%Y-%m-%dT%H:59:59Z')
    elif total_seconds >= 60:  # >= 1 minute but < 1 hour
        # Use minute granularity (YYYY-MM-DDTHH:MM:SSZ)
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:00Z')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:59Z')
    else:  # < 1 minute
        # Use second granularity (YYYY-MM-DDTHH:MM:SSZ)
        start_str = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Add date range to query
    query = f'{base_query} created:{start_str}..{end_str}'

    indent = "  " + "  " * depth
    print(f"{indent}Searching range {start_str} to {end_str}...")

    page = 1
    per_page = 100
    total_in_partition = 0

    while True:
        # Check debug limit
        if debug_limit is not None and total_in_partition >= debug_limit:
            print(f"{indent}  🐛 DEBUG MODE: Reached limit of {debug_limit} issues, stopping...")
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
            headers = token_pool.get_headers()
            response = request_with_backoff('GET', url, headers=headers, params=params)
            if response is None:
                print(f"{indent}  Error: retries exhausted for range {start_str} to {end_str}")
                return total_in_partition

            if response.status_code != 200:
                print(f"{indent}  Error: HTTP {response.status_code} for range {start_str} to {end_str}")
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
                print(f"{indent}  ⚠️ Hit 1000-result limit ({total_count} total). Splitting time range...")

                # Determine how to split based on time range duration
                if total_seconds < 2:  # Less than 2 seconds - can't split further
                    print(f"{indent}  ⚠️ Cannot split further (range < 2 seconds). Some results may be missing.")
                    break

                elif total_seconds < 120:  # Less than 2 minutes - split by seconds
                    # Split into 2-4 parts depending on range
                    num_splits = min(4, max(2, int(total_seconds / 30)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges (add 1 second to start)
                        if i > 0:
                            split_start = split_start + timedelta(seconds=1)

                        count = fetch_issues_with_time_partition(
                            base_query, split_start, split_end, token_pool, issues_by_id, debug_limit, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                elif total_seconds < 7200:  # Less than 2 hours - split by minutes
                    # Split into 2-4 parts
                    num_splits = min(4, max(2, int(total_seconds / 1800)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges (add 1 minute to start)
                        if i > 0:
                            split_start = split_start + timedelta(minutes=1)

                        count = fetch_issues_with_time_partition(
                            base_query, split_start, split_end, token_pool, issues_by_id, debug_limit, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                elif total_seconds < 172800:  # Less than 2 days - split by hours
                    # Split into 2-4 parts
                    num_splits = min(4, max(2, int(total_seconds / 43200)))
                    split_duration = time_diff / num_splits
                    split_dates = [start_date + split_duration * i for i in range(num_splits + 1)]

                    total_from_splits = 0
                    for i in range(num_splits):
                        split_start = split_dates[i]
                        split_end = split_dates[i + 1]
                        # Avoid overlapping ranges (add 1 hour to start)
                        if i > 0:
                            split_start = split_start + timedelta(hours=1)

                        count = fetch_issues_with_time_partition(
                            base_query, split_start, split_end, token_pool, issues_by_id, debug_limit, depth + 1
                        )
                        total_from_splits += count

                    return total_from_splits

                else:  # 2+ days - split by days
                    days_diff = time_diff.days

                    # Use aggressive splitting for large ranges or deep recursion
                    # Split into 4 parts if range is > 30 days, otherwise split in half
                    if days_diff > 30 or depth > 5:
                        # Split into 4 parts for more aggressive partitioning
                        quarter_diff = time_diff / 4
                        split_dates = [
                            start_date,
                            start_date + quarter_diff,
                            start_date + quarter_diff * 2,
                            start_date + quarter_diff * 3,
                            end_date
                        ]

                        total_from_splits = 0
                        for i in range(4):
                            split_start = split_dates[i]
                            split_end = split_dates[i + 1]
                            # Avoid overlapping ranges
                            if i > 0:
                                split_start = split_start + timedelta(days=1)

                            count = fetch_issues_with_time_partition(
                                base_query, split_start, split_end, token_pool, issues_by_id, debug_limit, depth + 1
                            )
                            total_from_splits += count

                        return total_from_splits
                    else:
                        # Binary split for smaller ranges
                        mid_date = start_date + time_diff / 2

                        # Recursively fetch both halves
                        count1 = fetch_issues_with_time_partition(
                            base_query, start_date, mid_date, token_pool, issues_by_id, debug_limit, depth + 1
                        )
                        count2 = fetch_issues_with_time_partition(
                            base_query, mid_date + timedelta(days=1), end_date, token_pool, issues_by_id, debug_limit, depth + 1
                        )

                        return count1 + count2

            # Normal pagination: check if there are more pages
            if len(items) < per_page or page >= 10:
                break

            page += 1
            time.sleep(0.5)  # Courtesy delay between pages

        except Exception as e:
            print(f"{indent}  Error fetching range {start_str} to {end_str}: {str(e)}")
            return total_in_partition

    if total_in_partition > 0:
        print(f"{indent}  ✓ Found {total_in_partition} issues in range {start_str} to {end_str}")

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
        'resolved_issues': resolved,
        'resolved_rate': round(resolved_rate, 2),
    }


def calculate_monthly_metrics_by_agent():
    """
    Calculate monthly metrics for all agents for visualization.
    Loads data directly from SWE-Arena/issue_metadata dataset.

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
    Uses batch folder upload to minimize commits (1 commit per agent instead of 1 per file).

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

    import tempfile
    import shutil

    temp_dir = None
    try:
        token = get_hf_token()
        if not token:
            raise Exception("No HuggingFace token found")

        api = HfApi()

        # Create temporary directory for batch upload
        temp_dir = tempfile.mkdtemp()
        agent_folder = os.path.join(temp_dir, agent_identifier)
        os.makedirs(agent_folder, exist_ok=True)

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        print(f"📤 Preparing batch upload for {agent_identifier} ({len(grouped)} daily files)...")

        for (issue_year, month, day), day_metadata in grouped.items():
            # New structure: [agent_identifier]/YYYY.MM.DD.jsonl
            filename = f"{agent_identifier}/{issue_year}.{month:02d}.{day:02d}.jsonl"
            local_filename = f"{issue_year}.{month:02d}.{day:02d}.jsonl"
            local_path = os.path.join(agent_folder, local_filename)

            print(f"   Preparing {len(day_metadata)} issues for {filename}...")

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

            # Save to temporary folder
            save_jsonl(local_path, merged_metadata)
            print(f"   ✓ Prepared {len(merged_metadata)} total issues for {local_filename}")

        # Upload entire folder in a single commit
        print(f"📤 Uploading folder {agent_identifier} to HuggingFace (1 commit)...")
        api.upload_folder(
            folder_path=agent_folder,
            path_in_repo=agent_identifier,
            repo_id=ISSUE_METADATA_REPO,
            repo_type="dataset",
            token=token,
            commit_message=f"Update metadata for {agent_identifier}"
        )
        print(f"   ✓ Successfully uploaded {len(grouped)} files in 1 commit")

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
    In debug mode, loads from in-memory cache if available.

    Structure: [agent_identifier]/YYYY.MM.DD.jsonl

    Returns:
        List of dictionaries with 'agent_identifier' added to each issue metadata.
        Only includes issues within the last LEADERBOARD_TIME_FRAME_DAYS.
    """
    # Calculate cutoff date based on LEADERBOARD_TIME_FRAME_DAYS
    current_time = datetime.now(timezone.utc)
    cutoff_date = current_time - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    # In debug mode, check in-memory cache first
    if DEBUG_MODE and DEBUG_ISSUE_METADATA_CACHE:
        all_metadata = []
        for agent_identifier, metadata_list in DEBUG_ISSUE_METADATA_CACHE.items():
            for issue_meta in metadata_list:
                # Filter by time frame in debug mode too
                created_at = issue_meta.get('created_at')
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        if dt < cutoff_date:
                            continue  # Skip issues outside time frame
                    except Exception:
                        pass  # Keep issues with unparseable dates

                issue_with_agent = issue_meta.copy()
                issue_with_agent['agent_identifier'] = agent_identifier
                all_metadata.append(issue_with_agent)
        if all_metadata:
            print(f"🐛 DEBUG MODE: Loading issue metadata from in-memory cache from last {LEADERBOARD_TIME_FRAME_DAYS} days ({len(all_metadata)} issues)")
            return all_metadata

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

        print(f"📥 Loading issue metadata from last {LEADERBOARD_TIME_FRAME_DAYS} days ({len(time_frame_files)} daily files across all agents)...")

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

def fetch_new_issues_for_agent(agent_identifier, token_pool, query_patterns=None, use_parallel=True):
    """
    Fetch and save new issues for an agent from yesterday 12am UTC to today 12am UTC.

    Args:
        agent_identifier: GitHub identifier of the agent
        token_pool: TokenPool instance for rotating tokens
        query_patterns: List of query patterns to search (if None, uses default)
        use_parallel: Whether to use parallel execution (default: True)

    Returns:
        Number of new issues found and saved
    """
    if not query_patterns:
        query_patterns = [
            'label:good-first-issue',
            'label:bug',
            'label:enhancement',
            'label:documentation',
        ]

    # Calculate time range: yesterday 12am UTC to today 12am UTC
    now_utc = datetime.now(timezone.utc)
    today_midnight = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_midnight = today_midnight - timedelta(days=1)

    print(f"\n  📥 Fetching new issues for {agent_identifier}...")
    print(f"     Time range: {yesterday_midnight.isoformat()} to {today_midnight.isoformat()}")

    total_new_issues = 0
    issues_by_id = {}

    # Add agent identifier to query patterns
    full_query_patterns = [f'author:{agent_identifier} {pattern}' for pattern in query_patterns]

    # Use parallel execution if enabled and multiple patterns exist and not in debug mode
    if use_parallel and len(full_query_patterns) > 1 and not DEBUG_MODE:
        try:
            total_new_issues = fetch_issues_parallel(
                full_query_patterns,
                yesterday_midnight,
                today_midnight,
                token_pool,
                issues_by_id,
                debug_limit=10 if DEBUG_MODE else None
            )
        except Exception as e:
            print(f"     ⚠️ Parallel execution failed, falling back to sequential: {str(e)}")
            use_parallel = False

    # Fall back to sequential if parallel is disabled or failed
    if not use_parallel or len(full_query_patterns) == 1 or DEBUG_MODE:
        for base_query in full_query_patterns:
            try:
                count = fetch_issues_with_time_partition(
                    base_query,
                    yesterday_midnight,
                    today_midnight,
                    token_pool,
                    issues_by_id,
                    debug_limit=10 if DEBUG_MODE else None,
                    depth=0
                )
                total_new_issues += count

            except Exception as e:
                print(f"     ⚠️ Error fetching pattern '{base_query}': {str(e)}")
                continue

    # Extract metadata from fetched issues
    if issues_by_id:
        metadata_list = [extract_issue_metadata(issue) for issue in issues_by_id.values()]

        # Save to HuggingFace
        success = save_issue_metadata_to_hf(metadata_list, agent_identifier)

        if success:
            print(f"  ✓ Saved {len(metadata_list)} new issues for {agent_identifier}")
        else:
            print(f"  ✗ Failed to save issues for {agent_identifier}")

    return total_new_issues


def update_all_agents_incremental():
    """
    Daily incremental update that:
    1. Refreshes all open issues from the last (LEADERBOARD_TIME_FRAME_DAYS - 1) days
       to check if they've been closed
    2. Fetches and adds new issues from yesterday 12am UTC to today 12am UTC

    Runs daily at 12:00 AM UTC as a scheduled task.
    """
    print(f"\n{'='*80}")
    print(f"🕛 Daily incremental mining started at {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*80}")

    try:
        # Load all GitHub tokens and create token pool
        tokens = get_github_tokens()
        token_pool = TokenPool(tokens)

        # Get first token for functions that still need single token
        token = tokens[0] if tokens else None

        # Load agent metadata from HuggingFace
        agents = load_agents_from_hf()
        if not agents:
            print("No agents found in HuggingFace dataset")
            return

        print(f"\n🔄 Phase 1: Refreshing open issues from last {LEADERBOARD_TIME_FRAME_DAYS - 1} days")
        print(f"   (checking if previously open issues have been closed)")

        total_checked = 0
        total_updated = 0

        # Step 1: Refresh all open issues from the last (LEADERBOARD_TIME_FRAME_DAYS - 1) days
        for agent in agents:
            identifier = agent.get('github_identifier')
            if not identifier:
                continue

            try:
                checked, updated = refresh_open_issues_for_agent(identifier, token)
                total_checked += checked
                total_updated += updated
            except Exception as e:
                print(f"   ⚠️ Error refreshing {identifier}: {str(e)}")
                continue

        print(f"\n   ✅ Phase 1 complete: {total_checked} open issues checked, {total_updated} updated")

        print(f"\n📥 Phase 2: Fetching new issues from yesterday 12am UTC to today 12am UTC")

        total_new_issues = 0

        # Step 2: Fetch new issues for each agent
        for agent in agents:
            identifier = agent.get('github_identifier')
            if not identifier:
                continue

            try:
                new_count = fetch_new_issues_for_agent(identifier, token_pool)
                total_new_issues += new_count
            except Exception as e:
                print(f"   ⚠️ Error fetching new issues for {identifier}: {str(e)}")
                continue

        print(f"\n   ✅ Phase 2 complete: {total_new_issues} new issues fetched")

        # Load updated metadata and calculate stats
        print(f"\n📊 Calculating updated statistics...")
        all_metadata = load_issue_metadata()

        for agent in agents:
            identifier = agent.get('github_identifier')
            agent_name = agent.get('agent_name', 'Unknown')

            if not identifier:
                continue

            try:
                # Filter metadata for this agent
                agent_metadata = [issue for issue in all_metadata if issue.get('agent_identifier') == identifier]

                # Calculate stats from metadata
                stats = calculate_issue_stats_from_metadata(agent_metadata)

                print(f"   ✓ {identifier}: {stats['total_issues']} issues, {stats['resolved_rate']}% resolved")

            except Exception as e:
                print(f"   ✗ Error processing {identifier}: {str(e)}")
                continue

        print(f"\n✅ Daily incremental mining completed at {datetime.now(timezone.utc).isoformat()}")

    except Exception as e:
        print(f"✗ Daily incremental mining failed: {str(e)}")
        import traceback
        traceback.print_exc()


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
    Construct leaderboard from issue metadata and convert to pandas DataFrame for display.
    Returns formatted DataFrame sorted by resolved rate.
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

    # Sort by Resolved Rate (%) descending
    if "Resolved Rate (%)" in df.columns and not df.empty:
        df = df.sort_values(by="Resolved Rate (%)", ascending=False).reset_index(drop=True)

    return df


def submit_agent(identifier, agent_name, organization, description, website):
    """
    Submit a new agent to the leaderboard.
    Validates input and saves submission. Issue data will be populated by daily incremental updates.
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

    return f"✅ Successfully submitted {agent_name}! Issue data will be populated by daily incremental updates.", get_leaderboard_dataframe(), create_monthly_metrics_plot()


# =============================================================================
# BACKGROUND TASKS
# =============================================================================


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

# Start APScheduler for daily regular issue mining at 12:00 AM UTC
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(
    update_all_agents_incremental,
    trigger=CronTrigger(hour=0, minute=0),  # 12:00 AM UTC daily
    id='daily_regular_mining',
    name='Daily Regular Issue Mining',
    replace_existing=True
)
scheduler.start()
print("✓ Scheduler started: Daily regular issue mining at 12:00 AM UTC")

# Create Gradio interface
with gr.Blocks(title="SWE Agent Issue Leaderboard", theme=gr.themes.Soft()) as app:

    gr.Markdown("# 🏆 SWE Agent Issue Leaderboard")
    gr.Markdown("Track and compare GitHub issue resolution statistics for SWE agents (last 6 months)")
    
    with gr.Tabs():
        
        # Leaderboard Tab
        with gr.Tab("📊 Leaderboard"):
            gr.Markdown("*All statistics are based on issues from the last 6 months*")
            leaderboard_table = Leaderboard(
                value=get_leaderboard_dataframe(),
                datatype=LEADERBOARD_COLUMNS,
                search_columns=["Agent Name", "Website"],
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