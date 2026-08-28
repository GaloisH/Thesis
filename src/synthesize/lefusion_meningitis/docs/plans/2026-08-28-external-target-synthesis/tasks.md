# Tasks: External Healthy Target Synthesis

**Spec**: [spec.md](spec.md)  
**Plan**: [plan.md](plan.md)

## Phase 1: Setup

- [x] T001 Inspect current configuration, synthesis pipeline, workspace changes, and applicable instructions. — Verify: inspected paths and dirty changes are recorded in the plan.

## Phase 2: Foundational

- [x] T002 Add `synthesis.target` path resolution in `config/lefusion_meningitis.yaml` and `src/synthesize/lefusion_meningitis/config.py`. — Verify: relative override resolves under project root.

## Phase 3: US1 — External healthy targets

**Goal**: Synthesize a lesion into an unlabeled flat-directory target.  
**Independent checkpoint**: One temporary mock case produces a QC-passed image and nonzero label.

- [x] T003 [US1] Add external target discovery and zero-label loading in `src/synthesize/lefusion_meningitis/synthesis/pipeline.py`. — Verify: invalid/empty directories fail before inference and a valid image yields its full-stem case ID.
- [x] T004 [US1] Run a temporary mock FastSurfer end-to-end test and delete its test script/data. — Verify: accepted_cases=1, accepted_lesions=1, background_exact=true, and no mock artifacts remain.

## Phase 4: US2 — Legacy compatibility

**Goal**: Preserve the existing public entry point and legacy path.  
**Independent checkpoint**: Default config without `synthesis.target` selects the prepared split path.

- [x] T005 [US2] Add stable dispatch between `synthesize_target` and `synthesize_legacy` in `src/synthesize/lefusion_meningitis/synthesis/pipeline.py`. — Verify: callable inspection and compile check show all three functions are available.

## Final Phase: Polish and Cross-cutting

- [x] T006 Review diffs, compile changed modules, update task evidence, and confirm unrelated dirty files are untouched. — Verify: scoped git diff and status inspection pass.

## Phase 5: Convergence

- [x] T007 Remove the stale `roi_from_mask` package export from `src/synthesize/lefusion_meningitis/synthesis/__init__.py`. — Verify: importing `lefusion_meningitis.synthesis` succeeds before the mock test starts.

## Validation Evidence

- Changed modules compiled successfully with `py_compile`.
- Temporary mock external-target synthesis passed with one accepted case, one accepted lesion, exact background preservation, a null source label, and full-stem case ID.
- Missing target directory failed before mocked runtime loading.
- Legacy dispatch was selected when `synthesis.target` was null.
- Temporary mock script and generated data were removed; IXI discovery found 581 direct NIfTI targets.
- Existing placement-geometry script passed 7/7 checks.
- `pytest` is unavailable in the active environment. The existing cortical-placement script reported 10/15 passing; its five failures use the repository's obsolete pre-`center_candidates` call signature and are unrelated to this feature.

## Dependencies and Parallelism

- T002 precedes T003/T005 because the target path must be resolved.
- T003 and T005 share `pipeline.py` and are implemented together, not in parallel.
- T004 depends on T002, T003, and T005.
- T006 depends on all implementation and validation tasks.

## Traceability

| Requirement / Scenario | Plan Decision | Tasks | Verification |
|---|---|---|---|
| FR-001 | target path contract | T002 | resolved-path assertion |
| FR-002, FR-003, FR-004 | TD-001, TD-002 | T003, T004 | mock external synthesis |
| FR-005 | shared dispatcher | T005 | default-config dispatch inspection |
| FR-006 | TD-003, TD-004 | T003, T004 | background and placement assertions |
| FR-007 | temporary validation strategy | T004, T006 | status/path inspection |
| NFR-001, NFR-002 | sorted discovery and preflight | T003, T004 | deterministic ID and invalid-path checks |
| NFR-003 | scoped patching | T001, T006 | final scoped diff |
| SC-001, SC-002, SC-003 | all decisions | T004, T006 | mock result and cleanup evidence |
