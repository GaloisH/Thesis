# Feature Specification: External Healthy Target Synthesis

**Feature ID**: external-target-synthesis  
**Created**: 2026-08-28  
**Status**: Implemented  
**Input**: Add one `synthesis.target` directory setting and synthesize lesions into unlabeled, non-nnUNet-named NIfTI images.

## Goal and Context

Allow a researcher to synthesize lesions into one flat directory of healthy T1 NIfTI volumes without creating nnUNet filenames or zero-label files first, while preserving the existing synthesis behavior when `synthesis.target` is absent.

## Scope

### In Scope

- Discover direct `*.nii.gz` children of `synthesis.target` in deterministic order.
- Use the filename without `.nii.gz` as the case ID.
- Create an all-zero aligned label in memory for every external target.
- Continue using the existing donor library, FastSurfer cortical constraint, sampling, QC, and output layout.
- Keep already inserted synthetic lesions mutually protected from overlap.
- Preserve the public `synthesize(config)` entry point and legacy behavior.

### Out of Scope

- Recursive discovery, configurable filename parsing, resampling, registration, automatic FastSurfer execution, export-policy changes, or support for target labels.
- Avoidance of unknown pre-existing lesions in external images.
- Retaining mock test code or mock data after validation.

## User Stories and Acceptance

### US1 — Synthesize into a healthy image directory (Priority: P1)

The researcher points `synthesis.target` at a flat NIfTI directory and receives synthetic images and labels without preparing target labels.

**Why P1**: This is the requested compatibility path.  
**Independent test**: Run one mocked external target through `synthesize(config)` with a mock aligned FastSurfer segmentation.

1. **Given** one external `.nii.gz` image, no target label, a donor patch, and an aligned cortical segmentation, **When** synthesis runs, **Then** one accepted synthetic case and a lesion label are produced.
2. **Given** the same input and seed, **When** synthesis runs twice, **Then** target discovery order and case ID are identical.
3. **Given** an absent or empty target directory, **When** synthesis starts, **Then** it fails before model inference with an actionable error.

### US2 — Preserve the legacy interface (Priority: P1)

Existing commands and callers continue to invoke `synthesize(config)` without changes.

**Why P1**: Existing experiments must remain reproducible.  
**Independent test**: Load the default configuration without `synthesis.target` and verify the legacy target-selection path is chosen.

1. **Given** no `synthesis.target`, **When** `synthesize(config)` is called, **Then** targets still come from the prepared split and nnUNet image/label paths.

## Edge Cases and Failure Behavior

- A target path that is not a directory is rejected.
- An empty directory is rejected.
- A filename that does not end in `.nii.gz` is not discovered.
- Missing FastSurfer segmentation remains a per-case skip under existing behavior.
- External images are not inspected for real lesions; their initial label is always zero.

## Requirements

### Functional

- **FR-001**: The system MUST resolve `synthesis.target` relative to the project root.
- **FR-002**: The system MUST dispatch to external-target synthesis when `synthesis.target` is configured.
- **FR-003**: The system MUST derive each external case ID from the full NIfTI filename stem.
- **FR-004**: The system MUST synthesize external targets without reading or requiring label files.
- **FR-005**: The system MUST preserve legacy synthesis when `synthesis.target` is absent.
- **FR-006**: The system MUST retain current FastSurfer placement, donor selection, QC, and output behavior.
- **FR-007**: Temporary mock test code and data MUST be removed after validation.

### Quality Attributes

- **NFR-001**: External target discovery MUST be deterministic.
- **NFR-002**: Failures for invalid target directories MUST occur before model loading.
- **NFR-003**: No existing unrelated workspace changes may be overwritten.

## Assumptions

- **A-001**: The target directory is flat and contains only intended T1 `.nii.gz` files.
- **A-002**: FastSurfer subject IDs equal the complete filename stem, including `-T1` for IXI.
- **A-003**: Target images have already received any desired spacing harmonization.

## Dependencies

- **D-001**: The existing donor manifest and donor patch files remain available.
- **D-002**: A FastSurfer segmentation and compatible LUT are available for each accepted case.

## Success Criteria

- **SC-001**: A temporary one-case external dataset with no label produces one accepted lesion and passes background-exact QC.
- **SC-002**: The output case ID equals the input filename without `.nii.gz`.
- **SC-003**: No temporary mock script or data remains in the repository after validation.
