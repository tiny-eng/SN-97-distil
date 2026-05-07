#!/usr/bin/env python3

# pip install huggingface_hub requests

# python compare_model_content_hash.py "repo-owner/model-a" "repo-owner/model-b"


from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import requests
from huggingface_hub import model_info, hf_hub_url


logger = logging.getLogger(__name__)


DEFAULT_TARGET_TENSORS = [
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
]


@dataclass
class ContentHashResult:
    repo: str
    revision: Optional[str]
    content_hash: Optional[str]
    tensor_hashes: List[str]
    missing_tensors: List[str]
    error: Optional[str] = None


def _auth_headers(token: Optional[str]) -> Dict[str, str]:
    """
    Build request headers for Hugging Face access.

    token is optional for public repos.
    For private/gated repos, pass --token or set it in your wrapper.
    """
    headers = {
        "Accept-Encoding": "identity",
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def _get_range_bytes(
    session: requests.Session,
    url: str,
    start: int,
    end: int,
    timeout: int,
) -> bytes:
    """
    Download an exact byte range from a remote file.

    This function requires HTTP 206 Partial Content so we know the server
    honored the Range request. That avoids accidentally hashing the wrong
    bytes if a server ignores Range and returns the whole file from byte 0.
    """
    if end < start:
        raise ValueError(f"Invalid byte range: {start}-{end}")

    headers = {
        "Range": f"bytes={start}-{end}",
        "Accept-Encoding": "identity",
    }

    response = session.get(
        url,
        headers=headers,
        timeout=timeout,
        stream=True,
        allow_redirects=True,
    )

    try:
        response.raise_for_status()

        if response.status_code != 206:
            raise RuntimeError(
                f"Server did not honor Range request. "
                f"Expected HTTP 206, got {response.status_code} for range {start}-{end}"
            )

        expected_size = end - start + 1
        data = response.raw.read(expected_size)

        if len(data) != expected_size:
            raise RuntimeError(
                f"Incomplete range read. Expected {expected_size} bytes, got {len(data)}"
            )

        return data

    finally:
        response.close()


def compute_content_hash(
    model_repo: str,
    revision: Optional[str] = None,
    sample_tensors: int = 4,
    token: Optional[str] = None,
    max_tensor_bytes: int = 200_000_000,
) -> ContentHashResult:
    """
    Compute a shard-invariant content hash for a Hugging Face model.

    It hashes raw bytes from selected tensors inside .safetensors files.
    If the same model is re-sharded, file-level SHA256 hashes may change,
    but tensor bytes remain identical.

    Parameters
    ----------
    model_repo:
        Hugging Face repo ID, e.g. "Qwen/Qwen2.5-0.5B".
    revision:
        Optional branch, tag, or commit hash.
    sample_tensors:
        Number of default target tensors to use.
    token:
        Optional Hugging Face token for private/gated repos.
    max_tensor_bytes:
        Skip tensors larger than this number of bytes.

    Returns
    -------
    ContentHashResult
        Includes final hash, per-tensor hashes, missing tensors, and error.
    """
    target_list = DEFAULT_TARGET_TENSORS[:sample_tensors]
    targets = set(target_list)
    found_tensor_hashes: List[str] = []

    try:
        info = model_info(
            model_repo,
            revision=revision,
            files_metadata=True,
            token=token,
        )

        st_files = sorted(
            s.rfilename
            for s in (info.siblings or [])
            if s.rfilename.endswith(".safetensors")
        )

        if not st_files:
            return ContentHashResult(
                repo=model_repo,
                revision=revision,
                content_hash=None,
                tensor_hashes=[],
                missing_tensors=target_list,
                error="No .safetensors files found",
            )

        with requests.Session() as session:
            session.headers.update(_auth_headers(token))

            for fname in st_files:
                if not targets:
                    break

                url = hf_hub_url(
                    repo_id=model_repo,
                    filename=fname,
                    revision=revision,
                )

                # Safetensors layout:
                # [8-byte little-endian header size][JSON header][raw tensor bytes]
                prefix = _get_range_bytes(
                    session=session,
                    url=url,
                    start=0,
                    end=7,
                    timeout=30,
                )

                if len(prefix) != 8:
                    continue

                header_size = struct.unpack("<Q", prefix)[0]

                if header_size <= 0 or header_size > 8_000_000:
                    logger.warning(
                        "Skipping %s because header size looks invalid: %s",
                        fname,
                        header_size,
                    )
                    continue

                header_len = 8 + header_size

                header_blob = _get_range_bytes(
                    session=session,
                    url=url,
                    start=0,
                    end=header_len - 1,
                    timeout=60,
                )

                header_json = json.loads(header_blob[8:header_len].decode("utf-8"))

                for tensor_name, tensor_info in header_json.items():
                    if tensor_name == "__metadata__":
                        continue

                    if tensor_name not in targets:
                        continue

                    offsets = tensor_info.get("data_offsets") or [0, 0]

                    if len(offsets) != 2:
                        logger.warning(
                            "Skipping %s in %s because data_offsets is invalid: %s",
                            tensor_name,
                            fname,
                            offsets,
                        )
                        continue

                    # Safetensors data_offsets are relative to the start of the
                    # tensor data section, which begins after the header.
                    abs_start = header_len + offsets[0]
                    abs_end = header_len + offsets[1] - 1

                    if abs_end < abs_start:
                        continue

                    tensor_size = abs_end - abs_start + 1

                    if tensor_size <= 0:
                        continue

                    if tensor_size > max_tensor_bytes:
                        logger.warning(
                            "Skipping tensor %s because it is too large: %.2f MB",
                            tensor_name,
                            tensor_size / 1_000_000,
                        )
                        continue

                    tensor_bytes = _get_range_bytes(
                        session=session,
                        url=url,
                        start=abs_start,
                        end=abs_end,
                        timeout=120,
                    )

                    tensor_sha = hashlib.sha256(tensor_bytes).hexdigest()
                    found_tensor_hashes.append(f"{tensor_name}:{tensor_sha}")
                    targets.discard(tensor_name)

        if not found_tensor_hashes:
            return ContentHashResult(
                repo=model_repo,
                revision=revision,
                content_hash=None,
                tensor_hashes=[],
                missing_tensors=target_list,
                error="No target tensors could be hashed",
            )

        found_tensor_hashes.sort()

        final_hash = hashlib.sha256(
            "\n".join(found_tensor_hashes).encode("utf-8")
        ).hexdigest()

        return ContentHashResult(
            repo=model_repo,
            revision=revision,
            content_hash=final_hash,
            tensor_hashes=found_tensor_hashes,
            missing_tensors=sorted(targets),
            error=None,
        )

    except Exception as exc:
        logger.warning("Content hash failed for %s: %s", model_repo, exc)

        return ContentHashResult(
            repo=model_repo,
            revision=revision,
            content_hash=None,
            tensor_hashes=found_tensor_hashes,
            missing_tensors=sorted(targets),
            error=str(exc),
        )


def compare_two_models(
    repo_a: str,
    repo_b: str,
    revision_a: Optional[str] = None,
    revision_b: Optional[str] = None,
    token: Optional[str] = None,
    sample_tensors: int = 4,
    max_tensor_bytes: int = 200_000_000,
) -> Tuple[ContentHashResult, ContentHashResult, bool]:
    """
    Compute content hashes for two Hugging Face repos and compare them.
    """
    result_a = compute_content_hash(
        model_repo=repo_a,
        revision=revision_a,
        sample_tensors=sample_tensors,
        token=token,
        max_tensor_bytes=max_tensor_bytes,
    )

    result_b = compute_content_hash(
        model_repo=repo_b,
        revision=revision_b,
        sample_tensors=sample_tensors,
        token=token,
        max_tensor_bytes=max_tensor_bytes,
    )

    is_match = (
        result_a.content_hash is not None
        and result_b.content_hash is not None
        and result_a.content_hash == result_b.content_hash
    )

    return result_a, result_b, is_match


def print_result(result: ContentHashResult, label: str) -> None:
    """
    Pretty-print one model's content-hash result.
    """
    print(f"\n=== {label} ===")
    print(f"Repo:      {result.repo}")
    print(f"Revision:  {result.revision or 'default'}")
    print(f"Hash:      {result.content_hash or 'None'}")

    if result.error:
        print(f"Error:     {result.error}")

    if result.tensor_hashes:
        print("\nTensor hashes:")
        for item in result.tensor_hashes:
            tensor_name, tensor_hash = item.split(":", 1)
            print(f"  - {tensor_name}: {tensor_hash}")

    if result.missing_tensors:
        print("\nMissing or skipped target tensors:")
        for tensor_name in result.missing_tensors:
            print(f"  - {tensor_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two Hugging Face models using shard-invariant tensor content hashes."
    )

    parser.add_argument(
        "repo_a",
        help='First Hugging Face repo, e.g. "user/model-a"',
    )

    parser.add_argument(
        "repo_b",
        help='Second Hugging Face repo, e.g. "user/model-b"',
    )

    parser.add_argument(
        "--revision-a",
        default=None,
        help="Optional revision/branch/tag/commit for first repo.",
    )

    parser.add_argument(
        "--revision-b",
        default=None,
        help="Optional revision/branch/tag/commit for second repo.",
    )

    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for private or gated repos.",
    )

    parser.add_argument(
        "--sample-tensors",
        type=int,
        default=4,
        help="Number of default target tensors to sample.",
    )

    parser.add_argument(
        "--max-tensor-bytes",
        type=int,
        default=200_000_000,
        help="Maximum tensor size to download/hash. Default: 200MB.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    result_a, result_b, is_match = compare_two_models(
        repo_a=args.repo_a,
        repo_b=args.repo_b,
        revision_a=args.revision_a,
        revision_b=args.revision_b,
        token=args.token,
        sample_tensors=args.sample_tensors,
        max_tensor_bytes=args.max_tensor_bytes,
    )

    print_result(result_a, "MODEL A")
    print_result(result_b, "MODEL B")

    print("\n=== COMPARISON ===")

    if result_a.content_hash is None or result_b.content_hash is None:
        print("Result: UNKNOWN")
        print("Reason: At least one model did not produce a valid content hash.")
    elif is_match:
        print("Result: MATCH")
        print("Meaning: The sampled tensor bytes are identical.")
        print("This strongly suggests the models are copies or re-sharded versions of the same weights.")
    else:
        print("Result: DIFFERENT")
        print("Meaning: The sampled tensor bytes are not identical.")
        print("This suggests the models are not exact copies under this tensor-hash check.")


if __name__ == "__main__":
    main()
