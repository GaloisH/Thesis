# Implementation Plan: External Healthy Target Synthesis

**Date**: 2026-08-28  
**Spec**: [spec.md](spec.md)  
**Status**: Implemented

## Summary

Add an external-target dispatcher around the existing synthesis implementation. External targets are discovered from one flat directory and represented as image paths with no label path; the shared synthesis body creates zero labels in memory. The legacy split-backed path remains the default.

## Technical Context

- Language/runtime: Python 3, PyTorch, NumPy, nibabel.
- Storage/state: NIfTI inputs and outputs plus JSON metadata.
- Testing: temporary one-case end-to-end mock using a generated NIfTI, donor NPZ, LUT, and FastSurfer MGH segmentation.
- Target platform: current Windows workspace and existing CLI.
- Performance/scale constraints: discovery must not load image voxel data; full synthesis cost is unchanged.
- Compatibility constraints: preserve `synthesize(config)` and existing YAML behavior.
- Known unknowns: none blocking; spacing harmonization is explicitly external to this feature.

## Governing Constraints Check — Pre-Design

| Constraint | Source | Status | Evidence/Action |
|---|---|---|---|
| Preserve unrelated dirty changes | AGENTS.md / workspace rules | Pass | Patch only adjacent lines and inspect diffs. |
| Complex work requires a plan | docs/SPEC.md | Pass | These artifacts precede application edits. |
| Mock test artifacts must be deleted | User request | Pass | Use a temporary directory and temporary script removed after execution. |
| Keep scope minimal | User request | Pass | No generalized adapter framework or resampling command. |

## Existing System Map

- `config/lefusion_meningitis.yaml`: runtime configuration.
- `config.py`: resolves configured paths.
- `synthesis/pipeline.py`: target selection, label loading, placement, sampling, QC, persistence.
- `synthesis/cortical_placement.py`: loads/resamples FastSurfer segmentation.
- `synthesis/sampling.py`: model sampling and hard composition.

## Proposed Design

### Architecture and Data Flow

`synthesize(config)` dispatches to `synthesize_target(config)` when `synthesis.target` is truthy and otherwise to `synthesize_legacy(config)`. Both construct target records and invoke one shared `_synthesize_cases()` body. A target record contains case ID, image path, and optional label path.

### Components and File Touchpoints

- Add one YAML field and one config path-resolution key.
- Refactor `pipeline.py` without changing sampling behavior.
- Do not change CLI because it already imports the stable wrapper.

### Data Model and Persistence

External labels are transient zero arrays. Output schemas remain compatible; metadata additionally records the source image and nullable source label.

### Interfaces and Contracts

- Public: `synthesize(config)` unchanged.
- New callable: `synthesize_target(config)`.
- Legacy callable: `synthesize_legacy(config)`.
- `synthesis.target`: flat directory of direct `.nii.gz` children.

### Error Handling and Observability

Reject missing, non-directory, or empty target paths before runtime/model loading. Preserve existing per-case FastSurfer failure records.

### Security, Privacy, Accessibility

Local paths only; no network or external disclosure. Accessibility is not applicable to this non-UI change.

## Technical Decisions

| ID | Decision | Alternatives | Rationale/Evidence | Consequence |
|---|---|---|---|---|
| TD-001 | Use filename stem as case ID | Regex adapter | Meets the one-line YAML constraint. | FastSurfer subject IDs must match the stem. |
| TD-002 | Create labels in memory | Write zero-label dataset | Avoids preprocessing files and extra interface. | Existing lesions cannot be protected. |
| TD-003 | Share one synthesis body | Duplicate pipeline | Prevents divergent sampling and QC logic. | Requires a bounded refactor of target loading. |
| TD-004 | Keep synthetic-to-synthetic overlap protection | Disable all protection | Avoids implausible overlap while honoring no pre-existing-lesion avoidance. | Later inserted lesions still avoid earlier inserted lesions. |

## Verification Strategy

- Compile changed modules.
- Verify configuration resolves a relative `synthesis.target`.
- Run a temporary end-to-end mock with one external image, no label, one donor patch, a mock cortical LUT/segmentation, and a mocked inference model.
- Assert one case and lesion are accepted, output label is nonzero, case ID matches filename stem, metadata source label is null, and background-exact QC passes.
- Remove the temporary script/data, then inspect status and diff.

## Delivery, Migration, and Rollback

No data migration. Removing `synthesis.target` restores legacy behavior. Rollback is limited to the new dispatcher/refactor and YAML/config key.

## Risks

| Risk | Likelihood/Impact | Mitigation | Detection |
|---|---|---|---|
| Filename/FastSurfer ID mismatch | Medium/High | Document full-stem rule. | Mock verifies expected segmentation path. |
| Geometry scale mismatch | High/Medium | Require target path to reference pre-harmonized data when needed. | Record source image and inspect spacing externally. |
| Refactor changes legacy behavior | Low/High | Keep target-record construction separate and shared body unchanged. | Compile plus legacy dispatch inspection. |
| Mock artifacts remain | Low/Medium | Temporary directory and explicit final filesystem check. | `git status` and path checks. |

## Governing Constraints Check — Post-Design

All pre-design constraints pass. The design adds only the requested directory mode, preserves the stable entry point, and gives every requirement a verification path.

## Complexity Exceptions

| Added Complexity | Why Necessary | Simpler Alternative Rejected Because |
|---|---|---|
| Shared target-record runner | A separate requested function must coexist with legacy behavior. | Duplicating the 300-line pipeline would create two maintenance paths. |
