"""
Minimalist Issue Metadata Mining Script
Mines issue metadata from GitHub and saves to HuggingFace dataset.
"""

import json
import os
import time
import requests
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

AGENTS_REPO = "SWE-Arena/swe_agents"
ISSUE_METADATA_REPO = "SWE-Arena/issue_metadata"
LEADERBOARD_TIME_FRAME_DAYS = 180  # 6 months

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
        import time
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
            from datetime import datetime, timezone
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


def get_hf_token():
    """Get HuggingFace token from environment variables."""
    token = os.getenv('HF_TOKEN')
    if not token:
        print("Warning: HF_TOKEN not found in environment variables")
    return token


# =============================================================================
# GITHUB API FUNCTIONS
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


def fetch_issues_parallel(query_patterns, start_date, end_date, token_pool, issues_by_id):
    """
    Fetch issues for multiple query patterns in parallel using available parallel tokens.

    Args:
        query_patterns: List of query patterns to search
        start_date: Start date for time range
        end_date: End date for time range
        token_pool: TokenPool instance for token management
        issues_by_id: Shared dictionary to store issues (thread-safe operations)

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
                pattern, start_date, end_date, token_pool, issues_by_id, depth=0
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


def fetch_issues_with_time_partition(base_query, start_date, end_date, token_pool, issues_by_id, depth=0):
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
        depth: Current recursion depth

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
                            base_query, split_start, split_end, token_pool, issues_by_id, depth + 1
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
                            base_query, split_start, split_end, token_pool, issues_by_id, depth + 1
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
                            base_query, split_start, split_end, token_pool, issues_by_id, depth + 1
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
                                base_query, split_start, split_end, token_pool, issues_by_id, depth + 1
                            )
                            total_from_splits += count

                        return total_from_splits
                    else:
                        # Binary split for smaller ranges
                        mid_date = start_date + time_diff / 2

                        # Recursively fetch both halves
                        count1 = fetch_issues_with_time_partition(
                            base_query, start_date, mid_date, token_pool, issues_by_id, depth + 1
                        )
                        count2 = fetch_issues_with_time_partition(
                            base_query, mid_date + timedelta(days=1), end_date, token_pool, issues_by_id, depth + 1
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

    Issue states:
    - state: "open" or "closed"
    - state_reason: "completed" (resolved), "not_planned" (closed as not planned), or None (still open)
    """
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


def fetch_all_issues_metadata(identifier, agent_name, token_pool, use_parallel=True):
    """
    Fetch issues associated with a GitHub user or bot for the past LEADERBOARD_TIME_FRAME_DAYS.
    Returns lightweight metadata instead of full issue objects.

    This function employs time-based partitioning to navigate GitHub's 1000-result limit per query.
    It searches using multiple query patterns:
    - is:issue author:{identifier} (issues authored by the bot)
    - is:issue assignee:{identifier} (issues assigned to the bot)

    Args:
        identifier: GitHub username or bot identifier
        agent_name: Human-readable name of the agent for metadata purposes
        token_pool: TokenPool instance for rotating tokens
        use_parallel: Whether to use parallel execution (default: True)

    Returns:
        List of dictionaries containing minimal issue metadata
    """

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

    # Define time range: past LEADERBOARD_TIME_FRAME_DAYS (excluding today)
    current_time = datetime.now(timezone.utc)
    end_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)  # 12:00 AM UTC today
    start_date = end_date - timedelta(days=LEADERBOARD_TIME_FRAME_DAYS)

    print(f"\n🔍 Fetching issues for {identifier}")
    print(f"   Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (today excluded)")
    print(f"   Query patterns: {len(query_patterns)}")

    total_start_time = time.time()

    # Use parallel execution if enabled and multiple patterns exist
    if use_parallel and len(query_patterns) > 1:
        try:
            print(f"\n   🚀 Using parallel execution for {len(query_patterns)} query patterns")
            total_found = fetch_issues_parallel(
                query_patterns,
                start_date,
                end_date,
                token_pool,
                issues_by_id
            )
        except Exception as e:
            print(f"   ⚠️ Parallel execution failed, falling back to sequential: {str(e)}")
            use_parallel = False

    # Fall back to sequential if parallel is disabled or failed
    if not use_parallel or len(query_patterns) == 1:
        for query_pattern in query_patterns:
            print(f"\n🔍 Searching with query: {query_pattern}")

            pattern_start_time = time.time()
            initial_count = len(issues_by_id)

            # Fetch with time partitioning
            issues_found = fetch_issues_with_time_partition(
                query_pattern,
                start_date,
                end_date,
                token_pool,
                issues_by_id
            )

            pattern_duration = time.time() - pattern_start_time
            new_issues = len(issues_by_id) - initial_count

            print(f"   ✓ Pattern complete: {new_issues} new issues found ({issues_found} total fetched, {len(issues_by_id) - initial_count - (issues_found - new_issues)} duplicates)")
            print(f"   ⏱️ Time taken: {pattern_duration:.1f} seconds")

            time.sleep(1.0)

    total_duration = time.time() - total_start_time
    all_issues = list(issues_by_id.values())

    print(f"\n✅ COMPLETE: Found {len(all_issues)} unique issues for {identifier}")
    print(f"   ⏱️ Total time: {total_duration:.1f} seconds")
    print(f"📦 Extracting minimal metadata...")

    metadata_list = [extract_issue_metadata(issue) for issue in all_issues]

    # Calculate memory savings
    import sys
    original_size = sys.getsizeof(str(all_issues))
    metadata_size = sys.getsizeof(str(metadata_list))
    savings_pct = ((original_size - metadata_size) / original_size * 100) if original_size > 0 else 0

    print(f"💾 Memory efficiency: {original_size // 1024}KB → {metadata_size // 1024}KB (saved {savings_pct:.1f}%)")

    return metadata_list


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


def upload_with_retry(api, path_or_fileobj, path_in_repo, repo_id, repo_type, token, max_retries=5):
    """
    Upload file to HuggingFace with exponential backoff retry logic.
    """
    delay = 2.0

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
                delay = min(delay * 2, 60.0)
            else:
                print(f"   ✗ Upload failed after {max_retries} attempts: {str(e)}")
                raise


def save_issue_metadata_to_hf(metadata_list, agent_identifier):
    """
    Save issue metadata to HuggingFace dataset, organized by [agent_identifier]/YYYY.MM.DD.jsonl.
    Each file is stored in the agent's folder and named YYYY.MM.DD.jsonl for that day's issues.

    This function APPENDS new metadata and DEDUPLICATES by html_url.
    Uses batch folder upload to minimize commits (1 commit per agent instead of 1 per file).

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

        api = HfApi()

        # Create temporary directory for batch upload
        temp_dir = tempfile.mkdtemp()
        agent_folder = os.path.join(temp_dir, agent_identifier)
        os.makedirs(agent_folder, exist_ok=True)

        # Group by exact date (year, month, day)
        grouped = group_metadata_by_date(metadata_list)

        print(f"📤 Preparing batch upload for {agent_identifier} ({len(grouped)} daily files)...")

        for (issue_year, month, day), day_metadata in grouped.items():
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
        return []


# =============================================================================
# MAIN MINING FUNCTION
# =============================================================================

def mine_all_agents():
    """
    Mine issue metadata for all agents within LEADERBOARD_TIME_FRAME_DAYS and save to HuggingFace.
    """
    # Load all GitHub tokens and create token pool
    tokens = get_github_tokens()
    token_pool = TokenPool(tokens)

    # Load agent metadata from HuggingFace
    agents = load_agents_from_hf()
    if not agents:
        print("No agents found in HuggingFace dataset")
        return

    print(f"\n{'='*80}")
    print(f"Starting issue metadata mining for {len(agents)} agents")
    print(f"Time frame: Last {LEADERBOARD_TIME_FRAME_DAYS} days")
    print(f"{'='*80}\n")

    # Mine each agent
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

            # Fetch issue metadata
            metadata = fetch_all_issues_metadata(identifier, agent_name, token_pool)

            if metadata:
                print(f"💾 Saving {len(metadata)} issue records...")
                save_issue_metadata_to_hf(metadata, identifier)
                print(f"✓ Successfully processed {agent_name}")
            else:
                print(f"   No issues found for {agent_name}")

        except Exception as e:
            print(f"✗ Error processing {identifier}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✅ Mining complete for all agents")
    print(f"{'='*80}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mine_all_agents()
