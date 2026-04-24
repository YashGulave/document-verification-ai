import json
import os
import urllib.error
import urllib.request
from collections import Counter


REFERENCE_DATASETS = [
    {
        "name": "DocTamper",
        "task": "tampered text detection in document images",
        "source": "https://github.com/qcf-568/DocTamper",
        "license_note": "Research dataset; repository notes non-commercial access requirements.",
    },
    {
        "name": "MIDV-500",
        "task": "identity document analysis and recognition from mobile video",
        "source": "https://huggingface.co/papers/1807.05786",
        "license_note": "Paper states source document images are public-domain or public-license material.",
    },
    {
        "name": "RVL-CDIP",
        "task": "document image classification and layout/domain reference",
        "source": "https://huggingface.co/datasets/aharley/rvl_cdip",
        "license_note": "Derived from IIT-CDIP / Legacy Tobacco Document Library; verify terms for your use case.",
    },
]

DATASET_CONTEXT = {
    "llm_provider": "Groq",
    "default_model": "llama-3.3-70b-versatile",
    "reference_dataset_names": [dataset["name"] for dataset in REFERENCE_DATASETS],
    "note": (
        "The prototype uses OCR, computer-vision evidence, and Groq reasoning, with public "
        "document-analysis datasets listed as domain references for evaluation and extension."
    ),
}

ISSUE_INTERPRETATIONS = {
    "edge_inconsistency": (
        "A marked area looks visually pasted or re-rendered because its text or border edges are much stronger "
        "than the rest of the document."
    ),
    "noise_variance_anomaly": (
        "A marked area has a different background texture from the surrounding page, which can happen when "
        "a field is edited, covered, copied from another image, or saved with different compression."
    ),
    "localized_blur_anomaly": (
        "A marked area is softer than nearby content, which can happen when an original value is erased, "
        "smoothed, retyped, or edited after capture."
    ),
    "trained_model_tamper_signal": (
        "The trained classifier found the overall document pattern similar to labeled tampered examples "
        "from the dataset used during model training."
    ),
}


def generate_explanation(
    issue_dicts: list[dict],
    ocr_text: str = "",
    ocr_status: str = "",
    *,
    groq_api_key: str | None = None,
    groq_model: str = "llama-3.3-70b-versatile",
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions",
    timeout_seconds: int = 20,
) -> tuple[str, str]:
    fallback_summary, fallback_json = _rule_based_explanation(issue_dicts)
    api_key = groq_api_key if groq_api_key is not None else os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return fallback_summary, fallback_json

    try:
        return _groq_explanation(
            issue_dicts,
            ocr_text,
            ocr_status,
            api_key=api_key,
            model=groq_model,
            api_url=groq_api_url,
            timeout_seconds=timeout_seconds,
            fallback_json=fallback_json,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, urllib.error.URLError) as error:
        fallback = json.loads(fallback_json)
        fallback["reasoning_source"] = "local_fallback"
        fallback["llm_error"] = _safe_error_message(error)
        return _fallback_summary_with_status(fallback_summary), json.dumps(fallback, indent=2)


def _rule_based_explanation(issue_dicts: list[dict]) -> tuple[str, str]:
    if not issue_dicts:
        data = {
            "verdict": "no_detected_tampering",
            "basis": [],
            "review_guidance": [
                "No unusual edit patterns were flagged by the current checks.",
                "If the document is high value, still compare it against the issuing authority's original records.",
            ],
            "reference_datasets": REFERENCE_DATASETS,
            "dataset_context": DATASET_CONTEXT,
            "reasoning_source": "local_rules",
            "limitations": "This result only means the implemented checks did not flag suspicious regions.",
        }
        return "The document does not show clear visual signs of field editing in the current scan.", json.dumps(data, indent=2)

    counts = Counter(issue["issue_type"] for issue in issue_dicts)
    max_confidence = max(issue["confidence"] for issue in issue_dicts)
    basis = []
    for issue in issue_dicts:
        issue_type = issue["issue_type"]
        basis.append(
            {
                "issue_type": issue_type,
                "confidence": issue["confidence"],
                "region": issue["box"],
                "measurement": issue["details"],
                "why_it_feels_suspicious": ISSUE_INTERPRETATIONS.get(
                    issue_type,
                    "This area differs from nearby document content and should be manually reviewed.",
                ),
                "specific_check": _specific_check_for_issue(issue_type),
            }
        )

    issue_names = ", ".join(f"{count} {_human_issue_name(kind).lower()}" for kind, count in sorted(counts.items()))
    summary = (
        f"This document needs review because the AI found {issue_names}. "
        f"The strongest suspicion score is {max_confidence:.2f}. "
        "The marked areas look inconsistent with the rest of the page and may indicate edited fields or inserted content."
    )
    data = {
        "verdict": "review_recommended",
        "highest_confidence": round(max_confidence, 3),
        "issue_counts": dict(counts),
        "document_specific_findings": _build_document_specific_findings(issue_dicts),
        "basis": basis,
        "review_guidance": _build_review_guidance(counts),
        "reference_datasets": REFERENCE_DATASETS,
        "dataset_context": DATASET_CONTEXT,
        "reasoning_source": "local_rules",
        "limitations": "This explanation uses detected OCR/CV evidence; it does not assert fraud by itself.",
    }
    return summary, json.dumps(data, indent=2)


def _groq_explanation(
    issue_dicts: list[dict],
    ocr_text: str,
    ocr_status: str,
    *,
    api_key: str,
    model: str,
    api_url: str,
    timeout_seconds: int,
    fallback_json: str,
) -> tuple[str, str]:
    evidence = {
        "issues": issue_dicts,
        "ocr_status": ocr_status,
        "ocr_excerpt": (ocr_text or "").strip()[:2500],
        "reference_datasets": REFERENCE_DATASETS,
        "local_fallback": json.loads(fallback_json),
    }
    payload = {
        "model": model,
        "temperature": 0.2,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You explain document verification evidence for a human reviewer. "
                    "Use only the supplied OCR and computer-vision evidence. Do not claim fraud, identity, "
                    "or authenticity as fact. Do not invent data sources or measurements. Mention the "
                    "provided open-source dataset references only as domain context for document analysis. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write like an AI document fraud reviewer, not like a computer vision log. "
                    "Explain why this specific document feels suspicious, which fields or zones should be checked, "
                    "and what practical verification steps the user should take. Avoid leading with words like "
                    "edge, blur, variance, OpenCV, or anomaly unless they appear inside the technical basis. "
                    "Create concise JSON with this shape: "
                    '{"summary": string, "structured": {"verdict": string, "highest_confidence": number|null, '
                    '"issue_counts": object, "document_specific_findings": array, "basis": array, "review_guidance": array, '
                    '"reference_datasets": array, "dataset_context": object, '
                    '"reasoning_source": "groq_llm", "limitations": string}}. '
                    f"Evidence: {json.dumps(evidence, ensure_ascii=True)}"
                ),
            },
        ],
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "HACTIVERSE-Document-Verifier/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read().decode("utf-8")

    completion = json.loads(raw)
    content = completion["choices"][0]["message"]["content"]
    generated = json.loads(content)
    summary = str(generated.get("summary", "")).strip()
    structured = generated.get("structured", {})
    if not summary or not isinstance(structured, dict):
        raise ValueError("Groq response did not include the expected explanation fields.")

    structured["reasoning_source"] = "groq_llm"
    structured.setdefault("reference_datasets", REFERENCE_DATASETS)
    structured["dataset_context"] = DATASET_CONTEXT
    structured.setdefault("document_specific_findings", _build_document_specific_findings(issue_dicts))
    structured.setdefault(
        "limitations",
        "This explanation is generated from OCR/CV signals and should be reviewed by a human before action.",
    )
    return summary, json.dumps(structured, indent=2)


def _build_review_guidance(counts: Counter) -> list[str]:
    guidance = []
    if counts.get("edge_inconsistency"):
        guidance.append("Zoom into the marked text or border area and compare it with nearby letters, stamps, lines, and field boxes.")
    if counts.get("noise_variance_anomaly"):
        guidance.append("Check whether the marked field has a different paper grain, background shade, or compression pattern.")
    if counts.get("localized_blur_anomaly"):
        guidance.append("Inspect whether softness is limited to important fields such as name, amount, date, ID number, or signature.")
    if counts.get("trained_model_tamper_signal"):
        guidance.append("Treat the trained-model signal as a document-level alert and use the highlighted CV regions to localize what to inspect.")
    guidance.append("Compare the suspicious fields with the original issuer record, payment record, or another copy of the same document.")
    return guidance


def _build_document_specific_findings(issue_dicts: list[dict]) -> list[str]:
    findings = []
    for issue in issue_dicts[:5]:
        box = issue.get("box", {})
        issue_type = issue.get("issue_type", "")
        confidence = issue.get("confidence", 0)
        findings.append(
            f"{_human_issue_name(issue_type)} around x={box.get('x')}, y={box.get('y')} "
            f"with suspicion score {confidence:.2f}: {_specific_check_for_issue(issue_type)}"
        )
    return findings


def _human_issue_name(issue_type: str) -> str:
    names = {
        "edge_inconsistency": "Possible pasted or retyped content",
        "noise_variance_anomaly": "Possible edited background field",
        "localized_blur_anomaly": "Possible erased or softened field",
        "trained_model_tamper_signal": "AI tamper-classifier signal",
    }
    return names.get(issue_type, issue_type.replace("_", " ").title())


def _specific_check_for_issue(issue_type: str) -> str:
    checks = {
        "edge_inconsistency": "Check whether the text style, border thickness, stamp edge, or field box looks inserted compared with nearby content.",
        "noise_variance_anomaly": "Check whether the paper background changes only around a value, name, date, signature, or amount.",
        "localized_blur_anomaly": "Check whether an important value was blurred, cleaned, or rewritten while the rest of the page stayed sharp.",
        "trained_model_tamper_signal": "Review the document as a whole because the trained classifier considers its visual pattern suspicious.",
    }
    return checks.get(issue_type, "Manually compare this area with the rest of the document and source records.")


def _fallback_summary_with_status(summary: str) -> str:
    return f"{summary} Groq reasoning was unavailable, so this explanation used the local evidence interpreter."


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        detail = error.read().decode("utf-8", errors="replace")[:500]
        return f"Groq HTTP {error.code}: {detail}"
    return f"{type(error).__name__}: {error}"
