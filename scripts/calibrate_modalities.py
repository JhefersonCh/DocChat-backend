"""Calibrate the relevance floor / threshold of each modality from labeled pairs (plan §3.5).

Cosines are only comparable inside one modality (unrelated image/text pairs already score
0.50-0.65), so every embedding profile carries per-modality {"floor", "threshold"}. This script
embeds labeled (query, media) pairs with the REAL model, computes per modality

  floor      = median cosine of the unrelated pairs        (relevance 0 at or below it)
  threshold  = cutoff of maximum F1 between matches / non-matches

and prints them; with --write it PATCHes them onto the embedding profile through the API
(recalibrating is editing the profile, not code; nothing else about the profile changes).

    uv run python scripts/calibrate_modalities.py scripts/calibration_pairs.json
    uv run python scripts/calibrate_modalities.py PAIRS.json --write [--profile UUID]

Needs the compose `embeddings` service (default http://127.0.0.1:8080/v1) and, for audio, ffmpeg.
--write also needs the API (default http://127.0.0.1:8000); without --profile it uses the ACTIVE
embedding profile. Pairs file (paths relative to the file itself; use real recorded voice for audio,
TTS is too clean to calibrate on):

    {"image": [{"query": "un perro", "file": "dog.jpg", "match": true},
               {"query": "un perro", "file": "cat.jpg", "match": false}],
     "audio": [{"query": "plazo de entrega", "file": "clip.wav", "match": true}]}

Aim for ~30 pairs per modality (balanced); the fit refuses to run without both classes.
"""

import argparse
import asyncio
import json
import statistics
from pathlib import Path

import httpx

from chat_pdf.adapters.audio.ffmpeg import FfmpegConverter
from chat_pdf.adapters.providers.openai_compatible import HttpProviderFactory
from chat_pdf.application.providers import LOCAL_EMBEDDING
from chat_pdf.domain.calibration import DEFAULT_CALIBRATION, MODALITIES, fit_calibration
from chat_pdf.domain.providers import ProviderProfile

MAX_AUDIO_SECONDS = 60


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
    return dot / norm if norm else 0.0


async def _score(modality: str, pairs: list[dict], base: Path, profile, factory) -> list[float]:
    text = factory.embedder(profile)
    media = factory.multimodal_embedder(profile)
    converter = FfmpegConverter(MAX_AUDIO_SECONDS)
    scores = []
    for pair in pairs:
        [query] = await text.embed([(profile.query_prefix or "") + pair["query"]])
        if modality == "image":
            vector = await media.embed_image((base / pair["file"]).read_bytes())
        elif modality == "audio":
            wav = await converter.to_wav16k((base / pair["file"]).read_bytes())
            vector = await media.embed_audio(wav)
        else:  # text: the passage is embedded as a document
            prefix = (profile.document_prefix or "").format(title="")
            [vector] = await text.embed([prefix + pair["text"]])
        scores.append(cosine(query, vector))
    return scores


def _report(modality: str, scores: list[float], labels: list[bool]) -> dict[str, float]:
    pos = [s for s, ok in zip(scores, labels, strict=True) if ok]
    neg = [s for s, ok in zip(scores, labels, strict=True) if not ok]
    fit = fit_calibration(scores, labels)
    spread = lambda xs: f"{min(xs):.3f}..{max(xs):.3f} (median {statistics.median(xs):.3f})"  # noqa: E731
    default = DEFAULT_CALIBRATION[modality]
    print(f"{modality}: {len(pos)} matches {spread(pos)} | {len(neg)} non-matches {spread(neg)}")
    print(
        f"   floor {fit.floor:.3f} (default {default.floor})  "
        f"threshold {fit.threshold:.3f} (default {default.threshold})"
    )
    return {"floor": fit.floor, "threshold": fit.threshold}


async def _run(args) -> None:
    data = json.loads(args.pairs.read_text())
    profile = ProviderProfile(
        id=None,  # type: ignore[arg-type]
        kind="embedding",
        base_url=args.embeddings_url,
        model=args.model,
        query_prefix=LOCAL_EMBEDDING["query_prefix"],
        document_prefix=LOCAL_EMBEDDING["document_prefix"],
        name="calibration",
    )
    factory = HttpProviderFactory(embed_batch=16)
    result = {}
    for modality in MODALITIES:
        pairs = data.get(modality) or []
        if not pairs:
            continue
        scores = await _score(modality, pairs, args.pairs.parent, profile, factory)
        result[modality] = _report(modality, scores, [bool(p["match"]) for p in pairs])
    if not result:
        raise SystemExit("the pairs file has no image / audio / text entries")
    print("\ncalibration =", json.dumps(result))
    if args.write:
        _write(args, result)


def _write(args, calibration: dict) -> None:
    with httpx.Client(base_url=args.api, timeout=30) as api:
        active = api.get("/api/v1/settings/active").json()["embedding_profile_id"]
        profile_id = args.profile or active
        if not profile_id:
            raise SystemExit("no embedding profile: pass --profile")
        stored = next(p for p in api.get("/api/v1/providers").json() if p["id"] == profile_id)
        merged = {**(stored.get("calibration") or {}), **calibration}  # keep other modalities
        res = api.patch(f"/api/v1/providers/{profile_id}", json={"calibration": merged})
        res.raise_for_status()
        print(f"written to profile {profile_id}: {res.json()['calibration']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("pairs", type=Path, help="labeled pairs JSON (see the module docstring)")
    parser.add_argument("--embeddings-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", default=LOCAL_EMBEDDING["model"])
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--profile", help="embedding profile UUID to write (default: the active)")
    parser.add_argument("--write", action="store_true", help="PATCH the result onto the profile")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
