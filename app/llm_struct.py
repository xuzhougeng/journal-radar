"""
LLM structured extraction client.
Uses OpenAI-compatible API to extract site type and summary from web content.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from app.config import get_runtime_config

logger = logging.getLogger(__name__)

# Valid site types (enum + other)
VALID_SITE_TYPES = {
    "paper",       # Academic paper / preprint
    "journal",     # Journal article page
    "news",        # News article
    "blog",        # Blog post
    "docs",        # Documentation / manual
    "repository",  # Code repository (GitHub, GitLab, etc.)
    "forum",       # Forum / discussion thread
    "product",     # Product page
    "dataset",     # Dataset description
    "other",       # None of the above
}


@dataclass
class StructuredResult:
    """Result from LLM structured extraction."""
    
    site_type: str  # One of VALID_SITE_TYPES
    site_type_reason: Optional[str]  # Reason if site_type is 'other'
    summary: str  # Generated summary
    raw_json: str  # Raw LLM response JSON
    model: str  # Model used
    response_time_ms: float  # Response time in milliseconds


def is_llm_enabled() -> bool:
    """Check if LLM structured extraction is enabled."""
    config = get_runtime_config()
    return bool(config.llm_api_key)


def truncate_text(text: Optional[str], max_chars: Optional[int] = None) -> str:
    """Truncate text to max characters."""
    if not text:
        return ""
    if max_chars is None:
        config = get_runtime_config()
        max_chars = config.llm_max_input_chars
    if len(text) <= max_chars:
        return text
    # Truncate with ellipsis
    return text[:max_chars - 3] + "..."


def build_prompt(title: Optional[str], url: Optional[str], text: Optional[str]) -> str:
    """
    Build the prompt for LLM structured extraction.
    
    Args:
        title: Page title from Exa
        url: Page URL
        text: Page content from Exa (will be truncated)
    
    Returns:
        The formatted prompt string.
    """
    config = get_runtime_config()
    truncated_text = truncate_text(text, config.llm_max_input_chars)
    
    site_types_list = ", ".join(sorted(VALID_SITE_TYPES - {"other"}))
    
    prompt = f"""Analyze the following web page and extract structured information.

**Page Information:**
- Title: {title or "N/A"}
- URL: {url or "N/A"}

**Page Content:**
{truncated_text or "No content available."}

**Task:**
1. Determine the site type from the following categories: {site_types_list}, or "other" if none fit.
2. If "other", provide a brief reason why it doesn't fit any category.
3. Write a concise summary (2-4 sentences) of the page content.

**Output Format:**
Return a JSON object with the following structure:
{{
    "site_type": "<one of the categories or 'other'>",
    "site_type_reason": "<reason if site_type is 'other', otherwise null>",
    "summary": "<2-4 sentence summary>"
}}

Only output the JSON object, no other text."""

    return prompt


async def call_chat_completions(
    messages: list[dict],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[int] = None,
) -> dict:
    """
    Call OpenAI-compatible chat completions API.
    
    Args:
        messages: List of message dicts with 'role' and 'content'.
        base_url: API base URL (defaults to config).
        api_key: API key (defaults to config).
        model: Model name (defaults to config).
        timeout: Request timeout in seconds (defaults to config).
    
    Returns:
        The full API response as a dict.
    
    Raises:
        httpx.HTTPError: If the request fails.
        ValueError: If the response is invalid.
    """
    config = get_runtime_config()
    
    base_url = base_url or config.llm_base_url
    api_key = api_key or config.llm_api_key
    model = model or config.llm_model
    timeout = timeout or config.llm_timeout
    
    if not api_key:
        raise ValueError("LLM API key not configured")
    
    # Build the API endpoint
    # Remove trailing slash from base_url if present
    base_url = base_url.rstrip("/")
    
    # Handle different base URL formats
    if "/v1/chat/completions" in base_url:
        endpoint = base_url
    elif "/v1" in base_url:
        endpoint = f"{base_url}/chat/completions"
    else:
        endpoint = f"{base_url}/v1/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,  # Lower temperature for more consistent outputs
        "response_format": {"type": "json_object"},  # Request JSON output
    }
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def parse_llm_response(response: dict) -> dict:
    """
    Parse and validate the LLM response.
    
    Args:
        response: The API response dict.
    
    Returns:
        Parsed dict with site_type, site_type_reason, and summary.
    
    Raises:
        ValueError: If parsing fails.
    """
    try:
        # Extract the assistant's message content
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No choices in response")
        
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise ValueError("No content in response")
        
        # Parse JSON from content
        # Try to extract JSON if there's extra text
        content = content.strip()
        if content.startswith("```"):
            # Remove markdown code block
            lines = content.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```"):
                    in_block = not in_block
                    continue
                if in_block or not line.startswith("```"):
                    json_lines.append(line)
            content = "\n".join(json_lines).strip()
        
        data = json.loads(content)
        
        # Validate site_type
        site_type = data.get("site_type", "other")
        if site_type not in VALID_SITE_TYPES:
            logger.warning(f"Invalid site_type '{site_type}', defaulting to 'other'")
            data["site_type_reason"] = f"LLM returned invalid type: {site_type}"
            site_type = "other"
        
        # Ensure site_type_reason is set for 'other'
        site_type_reason = data.get("site_type_reason")
        if site_type == "other" and not site_type_reason:
            site_type_reason = "Unspecified"
        elif site_type != "other":
            site_type_reason = None
        
        # Validate summary
        summary = data.get("summary", "")
        if not summary:
            summary = "No summary available."
        
        return {
            "site_type": site_type,
            "site_type_reason": site_type_reason,
            "summary": summary,
        }
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {e}")


async def extract_structured_info(
    title: Optional[str],
    url: Optional[str],
    text: Optional[str],
) -> StructuredResult:
    """
    Extract structured information (site type and summary) from web content.
    
    Args:
        title: Page title from Exa.
        url: Page URL.
        text: Page content from Exa.
    
    Returns:
        StructuredResult with extracted information.
    
    Raises:
        ValueError: If LLM is not configured or extraction fails.
    """
    if not is_llm_enabled():
        raise ValueError("LLM structured extraction is not configured")
    
    config = get_runtime_config()
    prompt = build_prompt(title, url, text)
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that analyzes web pages and extracts structured information. Always respond with valid JSON.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    
    start_time = time.time()
    
    try:
        response = await call_chat_completions(messages)
        response_time_ms = (time.time() - start_time) * 1000
        
        parsed = parse_llm_response(response)
        
        return StructuredResult(
            site_type=parsed["site_type"],
            site_type_reason=parsed["site_type_reason"],
            summary=parsed["summary"],
            raw_json=json.dumps(response, ensure_ascii=False),
            model=config.llm_model,
            response_time_ms=response_time_ms,
        )
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        raise


async def test_llm_connection() -> dict:
    """
    Test the LLM API connection with a simple request.
    
    Returns:
        Dict with test results including model and response time.
    
    Raises:
        ValueError: If LLM is not configured.
        httpx.HTTPError: If the request fails.
    """
    if not is_llm_enabled():
        raise ValueError("LLM API key not configured")
    
    config = get_runtime_config()
    
    messages = [
        {
            "role": "user",
            "content": 'Reply with exactly: {"status": "ok"}',
        },
    ]
    
    start_time = time.time()
    response = await call_chat_completions(messages)
    response_time_ms = (time.time() - start_time) * 1000
    
    return {
        "status": "success",
        "model": config.llm_model,
        "base_url": config.llm_base_url,
        "response_time_ms": round(response_time_ms, 2),
    }
