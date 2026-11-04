# Knowledge and content gap analysis

Status: implemented reference module; production activation is separate. This
design is part of the core knowledge system. Its human-facing status belongs in
the existing Notion system documentation, and its implementation belongs beside
this design in the repository. It does not create a second source catalogue,
knowledge index, review database, website or editorial planning system.

## Purpose and decision

Compare what authorized internal evidence and existing writing cover with
relevant external material and explicit questions or content plans. Return a
small review queue explaining what may be missing, why it matters, which exact
sources support the observation, what is uncertain, and a proposed next action.

A vector non-match cannot establish a knowledge gap. Keep these states distinct:

| Observation | Interpretation |
| --- | --- |
| Not accessible | The source cannot legitimately be read under the current grant |
| Not captured | The authorized collector has not supplied the source or has an unresolved interval |
| Not in the index | Captured material lacks a current eligible index representation |
| Not discussed in the observed scope | A bounded comparison found no supporting discussion; the rest of the corpus remains unknown |
| Proposed knowledge/content gap | An explicit question or plan has a source-grounded, reviewable shortfall |
| Validated gap | A human has assessed the shortfall and its scope through the existing review process |

The reference implementation emits only unvalidated proposals. It never turns
raw capture, similarity, task completion, a model conclusion or a Notion Status
into authenticated knowledge acceptance.

## Existing-system boundary

Existing passive collectors, source catalogue, extractors and native currentness
policies remain upstream. They own replay, source-specific reconciliation,
withdrawal/deletion, account access and downstream acknowledgement. Matching
watermarks or a healthy index do not prove source completeness.

Canonical retrieval owns current revision, tombstone, withholding, consent and
sensitivity filtering. The current PostgreSQL embedding profile remains the
retrieval baseline. Existing normalization behavior is preserved. In this design, the conservative
"isotropy" baseline is L2 normalization or a supplied fixed center followed by
L2; learned transforms remain an explicit experiment.

This module starts at the admitted evidence boundary:

1. An existing authorized reader selects a bounded source scope.
2. It provides normalized text, provenance, native revision and source state.
3. Deterministic screening and/or an existing authorized agent proposes gaps.
4. Mechanical validation binds citations to the admitted current text.
5. Full candidate-set diffs feed a small queue in the existing review surface.
6. A human may accept, revise or reject through the appropriate existing
   knowledge/task/editorial review mechanism.

No new inference endpoint or provider upload is embedded in this design. A thin
MCP wrapper can later call the same API; it must not add separate permission,
storage, source discovery or acceptance semantics. If an existing review contract admits only owner-authored knowledge, external
authors must not be relabeled as the owner to fit that schema. A general
claims/attribution extension needs its own reviewed contract.

## Source contract

The input schema is `hermes.gaps.input.v1`. Required fields are `schema_version`,
`as_of`, `scope` and `sources`. `as_of` is an explicit snapshot time, not a live
freshness guarantee. All supplied timestamps carry a UTC offset.

`scope` contains `account`, optional `tenant`, `policy_revision`,
`public_external_allowed` and `coverage` (`bounded`,
`complete_for_declared_scope` or `unknown`). The engine never upgrades an input
coverage assertion into a completeness finding. An admitted analysis domain is
not a general grant to combine upstream accounts; original account/workspace
identity stays in provenance and every reader must establish its own authority.

| Source field | Contract |
| --- | --- |
| `id`, `revision` | Stable logical identity and exact admitted revision; one record per source ID |
| `text`, `sha256` | Exact normalized UTF-8 text and its SHA-256; retain upstream hash, locator and projection mapping in provenance |
| `uri`, `title` | Reference metadata only; the engine never fetches or executes it |
| `kind` | `internal`, `external`, `own_draft`, `own_published` or `plan` |
| `account`, `tenant`, `policy_revision` | Match the admitted scope; only explicitly allowed public external records may use the public exception |
| `authorized`, `current`, `withdrawn` | Actual booleans asserted by the source reader |
| `access` | `accessible`, `not_accessible` or `unknown` |
| `capture` | `captured`, `not_captured` or `unknown` |
| `index` | `indexed`, `not_indexed` or `unknown` |
| `observed_at` | Required observation timestamp, never treated as publication time |
| `published_at` | Known publication timestamp or explicit `null`; cache, modification and draft-creation times are not publication dates |

Unauthorized, withdrawn, inaccessible and uncaptured bodies must be empty.
Noncurrent evidence is suppressed. Metadata-only source-state notices contain
no invented body quotes. A model cannot authorize a source by supplying these
fields: the engine verifies bindings, while the upstream reader authenticates
the assertions.

`text_basis` defaults to `canonical_index`. A source can instead use
`direct_current_source` when an existing authorized API, repository or document
store provides current text before canonical indexing. It must include
`projection: {reader, revision, text_sha256, checked_at}` with exact source
revision/hash and a check time no later than `as_of`. All permission/currentness
gates still apply. Index state stays truthful and produces a separate,
nonblocking notice; this path creates no parallel ingestion or index store.

Bounds are explicit: at most 256 sources, 200,000 characters per body,
2,000,000 total characters, 128 claim annotations and 128 topics. CLI inputs are
bounded regular JSON files, at most 16 MiB; duplicate keys, nonfinite numbers,
symlinks and FIFO inputs are rejected.

## Evidence and proposed meaning

Evidence uses Unicode codepoint offsets `[start,end)`, with a nonempty span of
at most 360 characters. Every output binds source ID, revision, full projection
hash, exact quote, quote hash, URI, text basis and index state. Offsets refer to
the normalized projection; the reader retains its mapping to the original
document. Metadata evidence uses the explicit empty range `[0,0)`.

Optional typed claims contain `id`, `subject`, `predicate`, `value`, `context`,
`valid_at`, `corroborated` and an evidence reference. Contradictions are proposed
only for different annotated values with the same subject/predicate/context
and effective instant. Different dates, scopes or embedding profiles do not
become simultaneous contradictions. Claim annotations are interpretations;
their text is not proved entailed just because the citation is exact.

Optional topics contain `id`, `title`, `terms`, `status`, `evidence` and an
`intent` of `learn` or `write`. Proposed topics stay visibly proposed. An
accepted topic requires a plan source plus a bound upstream human marker:
reviewer, review reference, source ID/revision and evidence hash. This checks
the received binding; it neither authenticates the human nor issues a grant.
No topic record creates an editorial commitment.

## Proposal engines

`hermes_cli/knowledge_gaps/core.py` provides inexpensive, inspectable screening:

| Category | Method | Important failure case |
| --- | --- | --- |
| Source coverage | Explicit access/capture/index/text states | A source limitation is not a knowledge gap |
| Unanswered question | Explicit question span without sufficient token overlap elsewhere | Rhetorical questions, paraphrases and omitted sources |
| Weak evidence | Claim annotation lacks recorded corroboration | Existing corroboration may be absent from the sample |
| Possible contradiction | Conflicting typed values in the same scope and time | Qualifiers or incorrect annotations can remove the conflict |
| Stale coverage | Known publication date exceeds a review horizon | Unknown dates skip this detector; age does not establish falsehood |
| External perspective | External text matches an explicit topic while internal/writing text does not | Lexical coverage is not understanding or full-corpus absence |
| Editorial opportunity | Evidence matches a writing target but sampled own writing does not | Missing inventory can explain the apparent opportunity |
| Redundant writing | Own draft and own published text are identical or share high shingle overlap | Intentional updates and syndication |

Internal and own-writing topic anchors count as real evidence; a plan anchor
alone does not. Ranking is a visible heuristic, not a calibrated probability.
There is no general "missing connection" inference from empty vector space.

`hermes_cli.knowledge_gaps.semantic.validate_comparisons()` accepts proposals from an already authorized
agent execution path. Each proposal includes a stable key, category, title,
observation, why it matters, next step, uncertainty, impact, `basis: agent_review`
and exact citations. Supported categories are `evidence_gap`,
`missing_connection`, `content_gap`, `possible_contradiction` and `redundancy`.
At least two distinct sources and two source kinds are required for this
cross-corpus path; same-kind internal claims use the typed-claim path.

Semantic citations must explicitly provide quote, revision and hash. Unknown
fields, including acceptance or execution state, are refused. This validator
proves citation bindings, not the truth of the reasoning or the identity of
the agent. Source instructions remain evidence, never executable instructions.

## Outputs, diffs and evaluation

The deterministic result schema is `hermes.gaps.output.v1`; semantic results use
`hermes.gaps.semantic.v1`. Both retain all candidates and cap the review queue at
five. Stable IDs identify the same category/topic within a scope; material
digests bind the assertions and their evidence. Every result is
`proposal_only`/`unvalidated`. The core's validated-gap count stays zero.

Diffs compare complete candidate sets, including items below the queue limit.
They report `added`, `material_changed`, `no_longer_observed` and unchanged IDs.
Missing candidates are never automatically resolved: withdrawal, sampling,
access or detector changes can remove them. Diffs do not repeat old quotes or
assertions. Cross-account/tenant comparisons and backward semantic-diff time
are refused. Detector, policy and coverage changes must remain separate from a
pure content change; refitting a transform is a mechanism change, not a new
knowledge discovery.

Snapshot hashes detect corruption, not forged authority. Queue/count,
proposal-only state and evidence bindings are checked before evaluation.
Labels bind the exact snapshot and declare `human`, `provisional` or
`synthetic` provenance. Unjudged items stay unjudged; complete-queue precision
is null while any item lacks judgment. An agent's self-label is never human
acceptance.

JSON goes to stdout or an existing consumer by default. Explicit machine
snapshots use private directories/files, fsync and atomic no-replace
publication; identical retries are no-ops. HTML reports and parallel review
pages are not an output path. Source review remains in existing Notion/GitHub
documentation and chat, with changes routed to the established human loop.

## Embedding geometry experiment

Geometry, retrieval and gap validity are separate evaluation questions.
Centering or whitening is an experimental transformation of a recoverable copy,
not an automatic replacement for the active embedding/index stack.

`hermes_cli/knowledge_gaps/experiments/geometry_benchmark.py` consumes cached vectors under
`hermes.geometry-benchmark.v1`, with declared model/dimension/profile hash,
source-family group, fit/evaluation split and explicit query relevance labels.
The sampler must authenticate source eligibility and preserve account and
deletion boundaries first. Profile declarations alone are not attestation.

The baseline is exact cosine over the same bounded evaluation candidates.
Centering fits only the training mean. PCA-only is an ablation for rank loss;
regularized whitening uses the same fitted basis and supported covariance rank.
The returned coordinate count stays unchanged while reduced intrinsic rank and
discarded directions are reported. Original vectors are never mutated.

Training source families cannot occur in evaluation documents or queries.
Exact vector duplicates crossing the split and accidental self-retrieval are
rejected. Related revisions and near-duplicate source families must be grouped
by the sampler. A bad transformed row skips the entire candidate rather than
silently removing a difficult case. Size bounds precede matrix allocation.

Measurements include Recall, direct-evidence Recall, MRR, pooled nDCG, judged
denominators, top-k hub counts, norms, pair-cosine distribution, covariance
concentration and effective rank. Weak adjacent-chunk labels remain separate
from human or provisional semantic judgments. No-positive cases have undefined
relevance metrics, not a fabricated success or failure.

Synthetic fixtures establish transform reconstruction, no input mutation,
source-family separation, handling of invalid/degenerate vectors and explicit
weak-label accounting. They do not establish a retrieval improvement. Actual
source-specific evaluation belongs in the authorized private review workflow;
source bodies, fitted transforms and live provenance are not committed here.

The research basis is mixed: [All-but-the-Top](https://arxiv.org/abs/1702.01417)
and [Whitening Sentence Representations](https://arxiv.org/abs/2103.15316)
motivate experiments, while [fine-tuned representation findings](https://aclanthology.org/2021.findings-emnlp.261/)
and [I-STAR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6691c5e4a199b72dffd9c90acb63bcd6-Abstract-Conference.html)
caution against treating greater isotropy as inherently better. Retrieval
and task evaluation should follow held-out evidence, informed by
[BEIR](https://arxiv.org/abs/2104.08663) and [MTEB](https://arxiv.org/abs/2210.07316).

## Generalizing to other sources

The core contains no author, publisher, source host or connector-specific
branches. HTML/Markdown extraction, OCR, transcripts and mail normalization
belong to existing extractors. An adapter supplies the same exact text and
source contract regardless of transport.

`hermes_cli.knowledge_gaps.adapters.collect()` validates all selected domains before the
first callback, freezes selections/readers/annotations, makes an owned snapshot
and rereads each selection. Identity, revision, policy or body drift refuses the
snapshot. Two observed reads are not atomic cross-system consistency.

Onboarding must prove: denied access causes zero reads; edits advance the
native revision; withdrawn evidence cannot support new proposals; exact quotes
map to the original source; same input reproduces; changed evidence creates a
material diff; and old citations are not served after a currentness failure.
Notion, repository Markdown, RSS, transcript and mail-shaped callbacks exercise
the same contract in tests. Those tests are not claims that five new production
connectors are deployed.

Do not add a manual-export dependency. Use existing passive capture or an
authorized native read API, keep original provider/account identity, and retain
explicit unresolved capture intervals. Public/subscribed access must be
legitimate; inaccessible paid material remains an access limitation.

## Integration and remaining production work

The reference implementation is testable independently and has no runtime
activation side effects. Existing collectors, databases, original embeddings,
tasks, content statuses and operational holds remain authoritative.

A live consumer still needs approved source adapters, current permission and
revision revalidation immediately before display, source-aware retention and
withdrawal, resource admission, independently judged semantic evaluation and
the appropriate existing review destination. Stored snapshots are not serving
authority; this module does not delete prior snapshots or bypass retention.

Before broader use, evaluate a larger source-family-held-out set of real
questions and plan comparisons, including negation, stale revisions, inaccessible
sources, missing capture, unrelated novelty, duplicate writing and intentional
syndication. Measure useful top-five precision, citation sufficiency, false
contradictions, false gap claims, redundant proposals and human review time.
Keep production embedding changes separate from accepting the gap module.

## CLI and reference examples

The same parser serves the public CLI and direct module invocation:

```sh
hermes knowledge-gaps validate --input tests/fixtures/knowledge_gaps/input.json
hermes knowledge-gaps analyze --input tests/fixtures/knowledge_gaps/input.json
hermes knowledge-gaps review --input tests/fixtures/knowledge_gaps/input.json \
  --proposals tests/fixtures/knowledge_gaps/proposals.json
python -m hermes_cli.knowledge_gaps validate --input tests/fixtures/knowledge_gaps/input.json
```

`diff --before before.json --after after.json` compares deterministic snapshots;
`evaluate --snapshot proposals.json --labels labels.json` scores only bound
judgments. `hermes_cli.knowledge_gaps.semantic.compare_semantic()` compares the
separate semantic report schema. Explicit `--output` is available when an
existing machine consumer needs an immutable private JSON snapshot.

The synthetic examples contain no real private corpus or source identifiers.
Tests live under `tests/hermes_cli/knowledge_gaps/` and use the canonical runner:

```sh
scripts/run_tests.sh -j 2 --file-retries 0 tests/hermes_cli/knowledge_gaps/ -q
```

The core and adapters require only the standard library. Geometry tests and the
optional geometry CLI require NumPy; ordinary commands never import it. The
reference module exposes no always-on model tool and changes no prompt/cache
state. A source adapter and the existing human-review consumer supply the
integration boundary.
