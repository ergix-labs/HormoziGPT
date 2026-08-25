from __future__ import annotations

from .store import SearchHit

SYSTEM_MESSAGE = """You are the Ergix Operator: a direct, evidence-grounded business coach.
Diagnose the constraint before prescribing. Prefer measurable actions over slogans.
Use the supplied knowledge as evidence, but never pretend to be Alex Hormozi or imply endorsement.
If the evidence does not support a claim, label it as your inference. Protect confidential data.
Every recommendation must name the move, the metric, a numeric target, and the timebox."""


def grounded_user_prompt(query: str, hits: list[SearchHit]) -> str:
    context = "\n\n".join(
        f"SOURCE {index}: {hit.title} ({hit.source})\n{hit.text}" for index, hit in enumerate(hits, 1)
    ) or "No matching internal evidence was found."
    return f"QUESTION\n{query}\n\nRETRIEVED KNOWLEDGE\n{context}\n\nAnswer directly and cite source titles when used."


DAILY_MOTIONS_PROMPT = """Create exactly {count} suggested motions for {date}.
Return JSON only with this shape:
{{"motions":[{{"title":"...","why_now":"...","action":"...","metric":"...","target":"...","timebox_minutes":30,"source_titles":["..."]}}]}}

Rules:
- Each motion must be executable today by one owner.
- Rank by expected business impact and current bottleneck, not novelty.
- Use a numeric target and a realistic 15-120 minute timebox.
- A target must never be zero. For a measurement motion, target one completed baseline.
- Do not invent business facts. If the profile is incomplete, recommend a measurement motion.
- Do not repeat substantially identical motions.
- `source_titles` may contain only titles copied exactly from RETRIEVED KNOWLEDGE.

BUSINESS PROFILE
{profile}

RETRIEVED KNOWLEDGE
{context}
"""
