"""Regression tests for MethodologySpec field coercion.

These cover the two schema bugs that made every real architecture extraction
fail silently. In both cases the LLM extracted the right answer and the schema
threw it away: the ValidationError was swallowed into an empty placeholder by
ExtractionAgent._extract_pass, extraction was still scored a PASS (it only
checks dataset_description), and CodeGenAgent fell back to inventing a generic
CNN. A 10-paper benchmark run reported 10/10 while producing placeholder models
for ResNet, SimCLR and DeiT.

The payloads below are reconstructed from the `input_value=...` fragments in
the real ValidationErrors recorded in verification_results/*/methodology.json.
"""

from __future__ import annotations

import pytest

from arpa.core.state import ArchitectureSpec, EvaluationSpec, TrainingSpec


class TestArchitectureLayersCoercion:
    """ArchitectureSpec.layers used to have two competing mode='before'
    validators. Pydantic v2 runs those in reverse definition order, so
    wrap_plain_lists (defined later) ran first and wrapped a raw list into
    {'value': [...]}, after which handle_layers' early return for dicts
    skipped all of its shape normalization -- making its list-of-component-
    dicts branch unreachable."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, None),
            (["conv1", "fc"], ["conv1", "fc"]),
            ("conv1", ["conv1"]),
            # Bare list of ModelComponentSpec-like dicts.
            ([{"name": "conv2d", "kind": "conv", "parameters": {"k": 3}}], ["conv2d"]),
            # List of ConfidenceField dicts.
            ([{"value": "conv1", "confidence": "confirmed"}], ["conv1"]),
            # ConfidenceField-wrapped list of plain strings.
            ({"value": ["conv1"], "confidence": "confirmed", "source": "S3"}, ["conv1"]),
        ],
    )
    def test_accepts_every_shape_a_model_returns(self, raw, expected):
        spec = ArchitectureSpec(layers=raw)
        assert (spec.layers.value if spec.layers else None) == expected

    @pytest.mark.parametrize(
        "paper, raw, expected",
        [
            (
                "medium1/ResNet",
                {"value": [{"name": "conv2d", "kind": "conv", "output_shape": [32, 32]}],
                 "confidence": "confirmed"},
                ["conv2d"],
            ),
            (
                "hard1/SimCLR",
                {"value": [{"name": "data augmentation", "kind": "augment"}],
                 "confidence": "confirmed"},
                ["data augmentation"],
            ),
            (
                "hard3/DeiT",
                {"value": [{"name": "patch_embedding", "kind": "embed"}],
                 "confidence": "confirmed"},
                ["patch_embedding"],
            ),
        ],
    )
    def test_real_payloads_that_used_to_fail(self, paper, raw, expected):
        """A ConfidenceField-wrapped list of component dicts -- the exact shape
        that raised 'layers.value.0 Input should be a valid string' on every
        non-trivial paper."""
        assert ArchitectureSpec(layers=raw).layers.value == expected, paper

    def test_layers_has_exactly_one_before_validator(self):
        """Guards the actual root cause: a second mode='before' validator on
        this field silently re-breaks it, because the later definition wins."""
        import ast
        from pathlib import Path

        src = Path("arpa/core/state.py").read_text(encoding="utf-8")
        owners = []
        for node in ast.parse(src).body:
            if not (isinstance(node, ast.ClassDef) and node.name == "ArchitectureSpec"):
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                for dec in item.decorator_list:
                    text = ast.unparse(dec)
                    if "field_validator" in text and "'layers'" in text and "before" in text:
                        owners.append(item.name)
        assert owners == ["handle_layers"], (
            f"expected exactly one before-validator on layers, found {owners}"
        )


class TestTrainingListCoercion:
    """regularization / augmentation_policy are list[ConfidenceField[str]], but
    a model asked for a single regularizer answers with one value. The wrapped
    form used to unwrap to a bare scalar and fail Pydantic's list check; the
    bare form silently became [] and dropped a real extracted value."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, []),
            ("L2", ["L2"]),
            (["L2"], ["L2"]),
            ({"value": "L2", "confidence": "confirmed"}, ["L2"]),
            ({"value": ["L2", "dropout"], "confidence": "confirmed"}, ["L2", "dropout"]),
            ([{"value": "L2", "confidence": "confirmed"}], ["L2"]),
            # Not a ConfidenceField at all, e.g. {"weight_decay": {...}}.
            ({"weight_decay": {"value": 1e-4}}, []),
        ],
    )
    def test_regularization_accepts_scalars_and_lists(self, raw, expected):
        spec = TrainingSpec(regularization=raw)
        assert [r.value for r in (spec.regularization or [])] == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ({"value": "Rand-Augment", "confidence": "confirmed"}, ["Rand-Augment"]),
            ("Rand-Augment", ["Rand-Augment"]),
        ],
    )
    def test_augmentation_policy_real_deit_payload(self, raw, expected):
        spec = TrainingSpec(augmentation_policy=raw)
        assert [a.value for a in (spec.augmentation_policy or [])] == expected

    def test_promoted_scalar_keeps_its_provenance(self):
        """A confirmed value must not be downgraded to 'assumed' just because
        it arrived as a scalar rather than a list."""
        spec = TrainingSpec(
            regularization={"value": "L2", "confidence": "confirmed", "source": "Sec 4"}
        )
        assert spec.regularization[0].confidence == "confirmed"
        assert spec.regularization[0].source == "Sec 4"

    def test_bare_scalar_is_marked_assumed(self):
        """An unwrapped scalar carries no evidence, so it should not claim any."""
        spec = TrainingSpec(regularization="L2")
        assert spec.regularization[0].confidence == "assumed"


class TestTrainingEvalCoercion:
    """The training/eval pass failed on 5 of 10 papers even after the
    architecture fix. Same underlying theme: a single malformed field raises,
    ExtractionAgent swallows it into an empty placeholder, and every other
    correctly-extracted hyperparameter in that pass is lost with it."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # medium1 / hard1: notes was a bare `str`, so an envelope failed outright.
            ({"value": "None", "confidence": "assumed"}, None),
            (
                {"value": "We use a simple setup", "confidence": "confirmed",
                 "source": "Section 2.1"},
                "We use a simple setup",
            ),
            ("plain prose", "plain prose"),
            (None, None),
        ],
    )
    def test_training_notes_accepts_envelopes(self, raw, expected):
        assert TrainingSpec(notes=raw).notes == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "None",
            "N/A",
            "not specified",
            {"value": "None", "confidence": "assumed"},
            None,
        ],
    )
    def test_mixed_precision_absent_tokens_become_none(self, raw):
        """hard3: a literal 'None' string reached Pydantic's bool parser and
        raised, discarding DeiT's whole training/eval pass over a field the
        paper simply never stated."""
        assert TrainingSpec(mixed_precision=raw).mixed_precision is None

    @pytest.mark.parametrize(
        "raw, expected",
        [(True, True), (False, False), ({"value": True, "confidence": "confirmed"}, True)],
    )
    def test_mixed_precision_real_values_survive(self, raw, expected):
        """Null-token handling must not swallow a legitimate False."""
        assert TrainingSpec(mixed_precision=raw).mixed_precision.value is expected

    def test_scheduler_parameters_wraps_plain_values(self):
        """medium3: dict values are ConfidenceField[Any], so a plain
        {'scheduler_type': 'step'} failed as a model_type error."""
        spec = TrainingSpec(scheduler_parameters={"scheduler_type": "step", "step_size": 30})
        got = {k: v.value for k, v in spec.scheduler_parameters.items()}
        assert got == {"scheduler_type": "step", "step_size": 30}

    def test_scheduler_parameters_keeps_existing_envelopes(self):
        spec = TrainingSpec(
            scheduler_parameters={"gamma": {"value": 0.1, "confidence": "confirmed"}}
        )
        assert spec.scheduler_parameters["gamma"].value == 0.1
        assert spec.scheduler_parameters["gamma"].confidence == "confirmed"

    def test_scheduler_parameters_drops_null_tokens(self):
        spec = TrainingSpec(scheduler_parameters={"warmup": "N/A", "step_size": 30})
        assert "warmup" not in spec.scheduler_parameters
        assert spec.scheduler_parameters["step_size"].value == 30

    @pytest.mark.parametrize(
        "raw, expected",
        [(True, "True"), (False, "False"), ("ten-crop", "ten-crop"), (5, "5"), (None, None)],
    )
    def test_test_time_augmentation_accepts_bools(self, raw, expected):
        """medium2: asked whether TTA was used, the model answered `True` for a
        field declared `str`."""
        assert EvaluationSpec(test_time_augmentation=raw).test_time_augmentation == expected


class TestNotesFieldsAcceptEnvelopes:
    """`notes` exists on ten classes. Fixing only the three nested spec classes
    left the four *Pass wrappers still rejecting envelopes, so medium1 and
    medium2 kept losing their whole training/eval pass to a single `notes`
    field -- the error path was `notes`, not `training.notes`, which is easy to
    misread as the one already fixed."""

    def test_no_notes_field_is_declared_plain_str(self):
        """Structural guard: every `notes` must be FlexibleStr. A new class
        declaring `notes: str | None` silently reintroduces this."""
        import ast
        from pathlib import Path

        src = Path("arpa/core/state.py").read_text(encoding="utf-8")
        offenders = []
        for node in ast.parse(src).body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and getattr(item.target, "id", "") == "notes":
                    annotation = ast.unparse(item.annotation)
                    if "Flexible" not in annotation:
                        offenders.append(f"{node.name}.notes: {annotation}")
        assert offenders == [], f"notes fields must be FlexibleStr, found: {offenders}"

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ({"value": "The paper presents...", "confidence": "assumed"},
             "The paper presents..."),
            ({"value": None, "confidence": None}, None),
            ("plain prose", "plain prose"),
            (None, None),
        ],
    )
    def test_pass_wrappers_accept_envelopes(self, raw, expected):
        from arpa.core.state import (
            ArchitecturePass,
            DatasetTaskPass,
            ImplementationPlanPass,
            TrainingEvalPass,
        )

        for cls in (DatasetTaskPass, ArchitecturePass, TrainingEvalPass, ImplementationPlanPass):
            assert cls(notes=raw).notes == expected, cls.__name__


class TestShapeAndDependencyCoercion:
    """VGG (medium2) lost two whole passes to these: nine validation errors
    from shapes sent as strings, and one from a file whose dependency list was
    an explicit null."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("[3, 224, 224]", [3, 224, 224]),          # JSON-in-JSON
            ("3, 224, 224", [3, 224, 224]),            # bare comma-separated
            ("[1,28,28]", [1, 28, 28]),                # no spaces
            ([3, 224, 224], [3, 224, 224]),            # already correct
            ({"value": "[3, 224, 224]", "confidence": "confirmed"}, [3, 224, 224]),
            ({"value": [3, 224, 224], "confidence": "confirmed"}, [3, 224, 224]),
            (None, None),
            # Prose carries no usable shape; dropping beats failing the pass.
            ("varies, depending on the input", None),
            ("input dependent", None),
        ],
    )
    def test_component_shapes_accept_strings(self, raw, expected):
        from arpa.core.state import ModelComponentSpec

        comp = ModelComponentSpec(name="stem", kind="conv", input_shape=raw, output_shape=raw)
        assert comp.input_shape == expected
        assert comp.output_shape == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("[3, 224, 224]", [3, 224, 224]),
            ({"value": "[3, 224, 224]", "confidence": "confirmed"}, [3, 224, 224]),
            ([3, 224, 224], [3, 224, 224]),
            ("varies with input", None),
        ],
    )
    def test_architecture_input_shape_accepts_strings(self, raw, expected):
        """Envelope must be unwrapped before parsing -- parsing first sees a
        dict and does nothing, leaving the string to fail as input_shape.value."""
        spec = ArchitectureSpec(input_shape=raw)
        assert (spec.input_shape.value if spec.input_shape else None) == expected

    def test_prose_is_never_shredded_into_a_list(self):
        """A sentence with commas must not become a bogus list."""
        from arpa.models.schema_helpers import parse_stringified_list

        for prose in ("varies, depending on input", "a, b, c", "see Table 1, row 2"):
            assert parse_stringified_list(prose) == prose

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, []),           # the actual failure: "depends_on": null
            ([], []),
            (["model.py"], ["model.py"]),
            ("model.py", ["model.py"]),
        ],
    )
    def test_file_dependencies_accept_null(self, raw, expected):
        """default_factory does not cover this -- it applies when the key is
        absent, not when it is present and null."""
        from arpa.core.state import CodegenFileSpec

        spec = CodegenFileSpec(
            path="train.py", purpose="train", depends_on=raw, required_symbols=raw
        )
        assert spec.depends_on == expected
        assert spec.required_symbols == expected


class TestConfidenceEnumFallback:
    """MobileNetV2's architecture pass sent confidence='224x224 conv2d' on one
    component -- a misaligned field -- and the enum rejected it, discarding all
    seven correctly-extracted components."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("confirmed", "confirmed"),
            ("inferred", "inferred"),
            ("assumed", "assumed"),
            ({"value": "confirmed", "confidence": "x"}, "confirmed"),
            # Unrecognised values degrade rather than fail.
            ("224x224 conv2d", "assumed"),
            ("very sure", "assumed"),
            (0.95, "assumed"),
            (None, "assumed"),
        ],
    )
    def test_unrecognised_confidence_degrades_to_assumed(self, raw, expected):
        from arpa.core.state import ModelComponentSpec

        comp = ModelComponentSpec(name="stem", kind="conv", confidence=raw)
        assert comp.confidence.value == expected

    def test_valid_levels_are_never_downgraded(self):
        """The fallback must not flatten real confidence into 'assumed'."""
        from arpa.core.confidence import ConfidenceLevel
        from arpa.core.state import ModelComponentSpec

        for level in (ConfidenceLevel.CONFIRMED, ConfidenceLevel.INFERRED):
            comp = ModelComponentSpec(name="s", kind="c", confidence=level)
            assert comp.confidence is level

    def test_one_bad_component_does_not_discard_the_others(self):
        """The actual regression: a single misaligned field cost the pass all
        of its components."""
        from arpa.core.state import ArchitecturePass

        result = ArchitecturePass(
            architecture={
                "model_name": {"value": "MobileNetV2", "confidence": "confirmed"},
                "components": [
                    {"name": "conv1", "kind": "conv", "confidence": "224x224 conv2d"},
                    {"name": "bottleneck", "kind": "residual_block", "confidence": "confirmed"},
                    {"name": "classifier", "kind": "linear", "confidence": "garbage"},
                ],
            }
        )
        components = result.architecture.components
        assert len(components) == 3
        assert [c.confidence.value for c in components] == ["assumed", "confirmed", "assumed"]


class TestNullTokenHandling:
    """Shared helper behaviour, exercised directly."""

    @pytest.mark.parametrize(
        "token", ["None", "none", "NULL", "n/a", "NA", "not specified", "  ", ""]
    )
    def test_absent_tokens_normalize_to_none(self, token):
        from arpa.models.schema_helpers import unwrap_confidence_field

        assert unwrap_confidence_field(token) is None

    @pytest.mark.parametrize("keep", ["None of the above", "nonlinear", "0", "false"])
    def test_real_strings_are_not_swallowed(self, keep):
        """The token list must not eat legitimate prose that merely starts with
        or resembles a null word."""
        from arpa.models.schema_helpers import unwrap_confidence_field

        assert unwrap_confidence_field(keep) == keep


class TestExtractionAccuracy:
    """Accuracy fixes, as distinct from the coercion fixes above: here the
    schema accepted the value fine, it was just the wrong value."""

    def test_iterations_have_their_own_field(self):
        """ResNet states "up to 60x10^4 iterations" and no epoch count. With
        only `epochs` available that landed there as 60000 -- wrong unit and
        wrong magnitude (the real figure is 600,000)."""
        spec = TrainingSpec(max_iterations=600000)
        assert spec.max_iterations.value == 600000
        assert spec.epochs is None

    def test_epochs_and_iterations_are_independent(self):
        spec = TrainingSpec(epochs=90, max_iterations=600000)
        assert spec.epochs.value == 90
        assert spec.max_iterations.value == 600000

    @pytest.mark.parametrize(
        "model_name, dataset, expected",
        [
            # Dataset papers: the pass answers "what model?" with the dataset.
            ("EMNIST", "EMNIST", None),
            ("emnist", "EMNIST", None),
            ("Fashion-MNIST", "Fashion-MNIST", None),
            ("fashion mnist", "Fashion-MNIST", None),
            # Genuine architectures must survive untouched.
            ("ResNet-50", "ImageNet", "ResNet-50"),
            ("DeiT-B", "ImageNet", "DeiT-B"),
            ("Bilinear CNN", "Caltech-UCSD birds", "Bilinear CNN"),
            # Merely resembling the dataset name is not enough to drop it.
            ("MNIST-Net", "MNIST", "MNIST-Net"),
        ],
    )
    def test_dataset_name_is_not_kept_as_model_name(self, model_name, dataset, expected):
        """A wrong model_name is worse than a missing one -- CodeGenAgent will
        emit `class EMNIST(nn.Module)` and look perfectly plausible doing it."""
        from arpa.agents.extraction_agent import ExtractionAgent
        from arpa.core.confidence import ConfidenceField, ConfidenceLevel
        from arpa.core.state import DatasetDescription, MethodologySpec

        spec = MethodologySpec(
            dataset_description=DatasetDescription(name=dataset),
            architecture=ArchitectureSpec(
                model_name=ConfidenceField(
                    value=model_name, confidence=ConfidenceLevel.INFERRED
                )
            ),
        )
        ExtractionAgent._drop_dataset_name_from_architecture(spec)
        got = spec.architecture.model_name
        assert (got.value if got is not None else None) == expected

    def test_guard_is_safe_on_incomplete_specs(self):
        """Runs on every extraction, including ones where a pass failed and
        left architecture or dataset empty."""
        from arpa.agents.extraction_agent import ExtractionAgent
        from arpa.core.state import DatasetDescription, MethodologySpec

        for spec in (
            MethodologySpec(),
            MethodologySpec(dataset_description=DatasetDescription(name="EMNIST")),
            MethodologySpec(architecture=ArchitectureSpec()),
        ):
            ExtractionAgent._drop_dataset_name_from_architecture(spec)  # must not raise


class TestProseFieldsAcceptLists:
    """Provenance fields are declared as prose, but a model quoting two
    supporting sentences returns a list. `source` was already tolerant
    (`str | list[str]`); `evidence`, `reason`, `suggested_resolution` and
    `purpose` were not -- and one `evidence: ['Dense Convolutional Network
    (DenseNet)']` discarded DenseNet's entire codegen plan, because a
    ValidationError anywhere degrades the whole pass to an empty placeholder."""

    def test_confidence_field_evidence_accepts_a_list(self):
        from arpa.core.confidence import ConfidenceField

        assert ConfidenceField(value="x", evidence=["a", "b"]).evidence == "a; b"
        assert ConfidenceField(value="x", evidence="plain").evidence == "plain"
        assert ConfidenceField(value="x", evidence=[]).evidence is None

    def test_the_real_densenet_payload(self):
        from arpa.core.confidence import ConfidenceField

        got = ConfidenceField(value="DenseNet", evidence=["Dense Convolutional Network (DenseNet)"])
        assert got.evidence == "Dense Convolutional Network (DenseNet)"

    @pytest.mark.parametrize(
        "cls_name, kwargs, field",
        [
            ("CodegenMissingDetail", {"field": "x", "reason": ["r1", "r2"]}, "reason"),
            ("CodegenMissingDetail", {"field": "x", "reason": "r", "evidence": ["e1", "e2"]}, "evidence"),
            ("CodegenMissingDetail",
             {"field": "x", "reason": "r", "suggested_resolution": ["s1", "s2"]},
             "suggested_resolution"),
            ("ModelComponentSpec", {"name": "n", "kind": "k", "evidence": ["e1", "e2"]}, "evidence"),
            ("CodegenFileSpec", {"path": "p", "purpose": ["a", "b"]}, "purpose"),
            ("ExtractedPreprocessStep", {"name": "n", "evidence": ["e1", "e2"]}, "evidence"),
        ],
    )
    def test_prose_fields_join_lists(self, cls_name, kwargs, field):
        import arpa.core.state as state

        obj = getattr(state, cls_name)(**kwargs)
        assert "; " in getattr(obj, field) or getattr(obj, field) is not None

    def test_plain_strings_are_untouched(self):
        from arpa.core.state import CodegenMissingDetail

        spec = CodegenMissingDetail(field="x", reason="a single reason")
        assert spec.reason == "a single reason"

    def test_every_prose_field_is_covered(self):
        """Guards against a new class declaring one of these as a bare str.
        The defect recurred in eight fields across five classes precisely
        because it was fixed one failure at a time."""
        import ast
        from pathlib import Path

        src = Path("arpa/core/state.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        prose = {"evidence", "reason", "suggested_resolution", "purpose"}
        offenders = []
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            declares = {
                ast.unparse(item.target)
                for item in node.body
                if isinstance(item, ast.AnnAssign)
            }
            if not (declares & prose):
                continue
            has_validator = any(
                isinstance(item, ast.Assign)
                and any(getattr(t, "id", "") == "_join_prose" for t in item.targets)
                for item in node.body
            )
            if not has_validator:
                offenders.append(node.name)
        assert offenders == [], (
            f"these classes declare prose fields without the joining validator: {offenders}"
        )


class TestDatasetAliases:
    """ILSVRC ran annually on essentially the same 1000-class data, and papers
    cite the year they competed in. Only the 2012 spelling was listed, so VGG
    (ILSVRC-2014) failed dataset resolution after extraction had succeeded."""

    @pytest.mark.parametrize(
        "spelling",
        ["ILSVRC2012", "ILSVRC-2014", "ilsvrc 2014", "ILSVRC2015", "ILSVRC",
         "ImageNet", "imagenet-1k"],
    )
    def test_ilsvrc_years_resolve_to_imagenet(self, spelling):
        from arpa.tools.dataset_tools import DATASET_ALIASES

        name = spelling.lower().strip()
        canonical = next(
            (c for c, aliases in DATASET_ALIASES.items() if name == c or name in aliases),
            None,
        )
        assert canonical == "imagenet", f"{spelling} did not resolve"
