"""Conservative, opt-in BONES primary-actor normalization and caption inventory.

Only a clearly identified first singular actor is renamed. Other people, action
words, modifiers, and subsequent sentences are not globally rewritten. Role nouns
such as dancer/bodybuilder retain their meaning and are reported for review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re


ORIGINAL_POLICY = "original"
CAMERA_WEARER_POLICY = "main_actor_camera_wearer_v1"
SUPPORTED_POLICIES = frozenset((ORIGINAL_POLICY, CAMERA_WEARER_POLICY))

_GENERIC_ACTORS = frozenset(
    "person man woman individual character human subject male female boy girl "
    "actor participant performer".split()
)
_ROLE_ACTORS = frozenset(
    "dancer athlete bodybuilder player soldier worker runner gymnast boxer wrestler".split()
)
# Explicit modifiers retain their original spelling and order. Unknown phrases
# are left for review rather than interpreted as adjectives by a broad regex.
_MODIFIERS = frozenset(
    "injured upright standing crouched bent frustrated confused seated hunched "
    "lonely young old older elderly adult shy crouching drunk angry cautious "
    "courteous lone forward-facing scared proud strong squatting inverted dancing "
    "happy sad tired exhausted calm nervous confident excited bored relaxed "
    "slightly very physically visibly tall short male female middle-aged and".split()
)


def _alternatives(words: frozenset[str]) -> str:
    return "|".join(re.escape(word) for word in sorted(words, key=lambda word: (-len(word), word)))


_ACTOR_WORDS = _alternatives(_GENERIC_ACTORS | _ROLE_ACTORS)
_MODIFIER_PATTERN = rf"(?:(?:{_alternatives(_MODIFIERS)})\b(?:\s+|,\s*))*"
_ARTICLE_ACTOR = re.compile(
    rf"(?P<article>a|an|the)\s+(?P<modifiers>{_MODIFIER_PATTERN})"
    rf"(?P<actor>{_ACTOR_WORDS})\b(?P<possessive>['\u2019]s)?",
    re.IGNORECASE,
)
_BARE_ACTOR = re.compile(r"(?:he|she|someone|somebody)\b", re.IGNORECASE)
_CANONICAL_ACTOR = re.compile(
    rf"the\s+{_MODIFIER_PATTERN}camera\s+wearer\b", re.IGNORECASE
)
_INTRO = re.compile(
    r"(?:while|when|after|before|standing|walking|sitting|crouching|lying|kneeling|"
    r"holding|turning|facing|bending|looking|raising|lowering|starting|leaning)\b",
    re.IGNORECASE,
)
_HUMAN_WORD = re.compile(
    rf"\b(?:{_ACTOR_WORDS}|people|men|women|someone|somebody|friend|peer|partner|"
    r"opponent|he|she|they)\b",
    re.IGNORECASE,
)
_POSSESSIVE_BODY = re.compile(
    r"\s+(?:(?:left|right|both|upper|lower|entire|whole)\s+)*"
    r"(?:head|neck|body|torso|chest|back|pelvis|hips?|arms?|hands?|fingers?|"
    r"thumbs?|legs?|feet|foot|toes?|shoulders?|wrists?|knees?|ankles?|elbows?|"
    r"forearms?|movement|movements|posture|balance|gaze)\b",
    re.IGNORECASE,
)
_FOLLOWUP_ACTOR = re.compile(
    rf"[.!?]\s+(?:(?:a|an|the)\s+{_MODIFIER_PATTERN}(?:{_ACTOR_WORDS})\b|"
    r"(?:he|she|they)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CaptionNormalization:
    caption: str
    status: str
    reason: str
    original_subject: str = ""
    normalized_subject: str = ""


def normalize_bones_caption(caption: str) -> CaptionNormalization:
    """Rename an unambiguous primary actor without resolving coreference.

    Empty classifier-free captions remain empty. Possessive body references are
    supported, but ownership of another entity is not interpreted as the actor.
    A fronted clause is supported only when it has no human reference and the
    following comma introduces an explicit singular actor.
    """
    if not caption.strip():
        return CaptionNormalization(caption, "unchanged", "empty")
    start = len(caption) - len(caption.lstrip())
    sentence_start = True
    remainder = caption[start:]
    if _CANONICAL_ACTOR.match(remainder):
        return CaptionNormalization(caption, "unchanged", "already_canonical")
    article_match = _ARTICLE_ACTOR.match(remainder)
    bare_match = _BARE_ACTOR.match(remainder)
    if article_match is None and bare_match is None:
        comma = caption.find(",", start)
        prefix = caption[start:comma] if comma >= 0 else ""
        if 0 < len(prefix) <= 200 and _INTRO.match(prefix) and not _HUMAN_WORD.search(prefix):
            start = comma + 1
            while start < len(caption) and caption[start].isspace():
                start += 1
            sentence_start = False
            remainder = caption[start:]
            if _CANONICAL_ACTOR.match(remainder):
                return CaptionNormalization(caption, "unchanged", "already_canonical")
            article_match = _ARTICLE_ACTOR.match(remainder)
            bare_match = _BARE_ACTOR.match(remainder)
    if article_match is None and bare_match is None:
        reason = "number_ambiguous_pronoun" if re.match(r"they\b", remainder, re.I) else "unrecognized_or_subjectless"
        return CaptionNormalization(caption, "review", reason)

    match = article_match if article_match is not None else bare_match
    assert match is not None
    tail = remainder[match.end():]
    original_subject = match.group(0)
    if re.match(r"\s*(?:,\s*)?(?:and|or)\b", tail, re.I):
        return CaptionNormalization(caption, "review", "coordinated_subject", original_subject)
    if article_match is not None:
        actor = article_match.group("actor").lower()
        modifiers = article_match.group("modifiers")
        if actor in _ROLE_ACTORS:
            return CaptionNormalization(caption, "review", "role_noun_preserved", original_subject)
        modifier_words = re.findall(r"[A-Za-z-]+", modifiers.lower())
        if modifier_words and (modifier_words[0] == "and" or modifier_words[-1] == "and"):
            return CaptionNormalization(caption, "review", "ambiguous_modifiers", original_subject)
        possessive = article_match.group("possessive") or ""
        if possessive and not _POSSESSIVE_BODY.match(tail):
            return CaptionNormalization(caption, "review", "non_body_possessive", original_subject)
    else:
        modifiers = ""
        possessive = ""
        if re.match(r"(?:['\u2019]|\s+else\b)", tail, re.I):
            return CaptionNormalization(caption, "review", "pronoun_contraction_or_ownership", original_subject)

    article = "The" if sentence_start else "the"
    normalized_subject = f"{article} {modifiers}camera wearer{possessive}"
    normalized = caption[:start] + normalized_subject + tail
    followup = bool(_FOLLOWUP_ACTOR.search(tail))
    return CaptionNormalization(
        normalized,
        "normalized_review" if followup else "normalized",
        "followup_coreference_unresolved" if followup else "clear_primary_actor",
        original_subject,
        normalized_subject,
    )


def write_inventory_report(inventory_cache: Path, output_dir: Path) -> dict:
    """Apply the policy to cached caption counts, never to source manifests."""
    inventory = json.loads(inventory_cache.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "policy": CAMERA_WEARER_POLICY,
        "inventory_cache": str(inventory_cache),
        "source_manifests_modified": False,
        "motion_or_sampling_changes": False,
        "review_rows_are_retained": True,
        "splits": {},
    }
    examples = []
    example_subjects = set()
    with (output_dir / "review_required.jsonl").open("x") as review_file:
        for split, source in inventory.items():
            status_rows = Counter()
            reason_rows = Counter()
            changed_rows = 0
            changed_unique = 0
            for caption, count in source["captions"].items():
                result = normalize_bones_caption(caption)
                status_rows[result.status] += count
                reason_rows[result.reason] += count
                changed = result.caption != caption
                changed_rows += count if changed else 0
                changed_unique += int(changed)
                record = {
                    "split": split, "rows": count, "status": result.status,
                    "reason": result.reason, "original": caption,
                    "normalized": result.caption,
                }
                if "review" in result.status:
                    review_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                if changed and len(examples) < 40 and result.original_subject.lower() not in example_subjects:
                    examples.append(record)
                    example_subjects.add(result.original_subject.lower())
            report["splits"][split] = {
                "source": source["source"], "rows": source["rows"],
                "unique_captions": len(source["captions"]),
                "changed_rows": changed_rows, "changed_unique_captions": changed_unique,
                "changed_percent": 100.0 * changed_rows / max(1, source["rows"]),
                "status_rows": dict(status_rows), "reason_rows": dict(reason_rows),
            }
    with (output_dir / "normalization_report.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    with (output_dir / "normalization_examples.json").open("x") as handle:
        json.dump(examples, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_inventory_report(args.inventory_cache, args.out), indent=2), flush=True)


if __name__ == "__main__":
    main()
