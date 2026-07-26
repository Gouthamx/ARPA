"""CodeGen readiness checker - validates sufficiency, not just schema validity.

This module checks whether extracted methodology data is SUFFICIENT for code
generation, not just structurally valid. A field can pass Pydantic validation
(correct type, non-null) but still be too vague to generate code from.

Gap: Schema validation checks shape. This checks semantic usability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arpa.core.state import MethodologySpec

# Known architecture families that can be implemented
KNOWN_ARCHITECTURES = {
    "resnet", "vgg", "densenet", "mobilenet", "efficientnet",
    "vit", "vision transformer", "transformer",
    "alexnet", "squeezenet", "shufflenet", "inception",
    "cnn", "conv", "mlp", "fcn", "unet"
}

# Known optimizers with standard implementations
KNOWN_OPTIMIZERS = {
    "sgd", "adam", "adamw", "rmsprop", "adagrad",
    "adadelta", "adamax", "nadam", "radam", "lamb"
}

# Known loss functions
KNOWN_LOSSES = {
    "crossentropy", "cross_entropy", "bce", "binary_crossentropy",
    "mse", "mean_squared_error", "mae", "mean_absolute_error",
    "huber", "focal", "contrastive", "triplet", "nll"
}


def _extract_value(field) -> str | None:
    """Extract actual value from ConfidenceField or plain value."""
    if field is None:
        return None
    
    # Handle Pydantic models (ConfidenceField)
    if hasattr(field, 'value'):
        return str(field.value) if field.value is not None else None
    
    # Handle dict representation
    if isinstance(field, dict) and "value" in field:
        return str(field["value"]) if field["value"] is not None else None
    
    # Plain value
    return str(field) if field else None


def check_codegen_readiness(methodology: MethodologySpec) -> tuple[bool, list[str]]:
    """Check if methodology has SUFFICIENT detail for code generation.
    
    Returns:
        (is_ready, list_of_gap_descriptions)
        
    This checks SUFFICIENCY, not just presence. A field can pass Pydantic
    validation and still fail here if it's too vague to generate code from.
    
    Examples of gaps:
    - architecture="a neural network" (too vague)
    - optimizer="optimizer" (no specific algorithm)
    - learning_rate=None (missing critical hyperparameter)
    - dataset_description="image classification" (no concrete details)
    """
    gaps = []
    
    # Check architecture sufficiency
    if not methodology.architecture:
        gaps.append("architecture: missing entirely")
    else:
        arch_name = _extract_value(methodology.architecture.model_name)
        if not arch_name:
            gaps.append("architecture.model_name: missing")
        else:
            arch_lower = arch_name.lower()
            # Check if architecture is specific enough
            if len(arch_lower) < 3:
                gaps.append(f"architecture: too short to be specific -- got '{arch_name[:60]}'")
            elif not any(known in arch_lower for known in KNOWN_ARCHITECTURES):
                # Check if it at least describes layer structure
                has_layers = any(word in arch_lower for word in ["conv", "layer", "block", "fc", "linear", "attention"])
                has_numbers = any(char.isdigit() for char in arch_lower)
                
                if not (has_layers and has_numbers):
                    gaps.append(
                        f"architecture: unrecognized and lacks concrete layer description -- "
                        f"got '{arch_name[:80]}' (need architecture family name OR layer structure with counts)"
                    )
    
    # Check training configuration sufficiency
    if not methodology.training:
        gaps.append("training: missing entirely")
    else:
        training = methodology.training
        
        # Optimizer
        opt = _extract_value(training.optimizer)
        if not opt:
            gaps.append("training.optimizer: missing")
        else:
            opt_lower = opt.lower()
            if len(opt_lower) < 3:
                gaps.append(f"training.optimizer: too vague -- got '{opt}'")
            elif not any(known in opt_lower for known in KNOWN_OPTIMIZERS):
                # Check if it's a generic placeholder
                if opt_lower in ["optimizer", "optimiser", "opt", "unknown"]:
                    gaps.append(f"training.optimizer: generic placeholder, not a specific algorithm -- got '{opt}'")
        
        # Learning rate
        lr = _extract_value(training.learning_rate)
        if not lr or lr == "None":
            gaps.append("training.learning_rate: missing (required for training)")
        else:
            try:
                lr_float = float(lr)
                if lr_float <= 0 or lr_float > 1:
                    gaps.append(f"training.learning_rate: unrealistic value -- got {lr_float}")
            except (ValueError, TypeError):
                gaps.append(f"training.learning_rate: not a valid number -- got '{lr}'")
        
        # Batch size
        bs = _extract_value(training.batch_size)
        if not bs or bs == "None":
            gaps.append("training.batch_size: missing (required for data loading)")
        else:
            try:
                bs_int = int(float(bs))
                if bs_int < 1 or bs_int > 10000:
                    gaps.append(f"training.batch_size: unrealistic value -- got {bs_int}")
            except (ValueError, TypeError):
                gaps.append(f"training.batch_size: not a valid integer -- got '{bs}'")
        
        # Epochs
        epochs = _extract_value(training.epochs)
        if not epochs or epochs == "None":
            gaps.append("training.epochs: missing (required for training loop)")
        else:
            try:
                ep_int = int(float(epochs))
                if ep_int < 1 or ep_int > 10000:
                    gaps.append(f"training.epochs: unrealistic value -- got {ep_int}")
            except (ValueError, TypeError):
                gaps.append(f"training.epochs: not a valid integer -- got '{epochs}'")
        
        # Loss function (if present, check it's specific)
        loss = _extract_value(training.loss_function) if hasattr(training, 'loss_function') else None
        if loss and loss != "None":
            loss_lower = loss.lower()
            if not any(known in loss_lower for known in KNOWN_LOSSES):
                if loss_lower in ["loss", "loss function", "loss_fn", "criterion", "unknown"]:
                    gaps.append(f"training.loss_function: generic placeholder -- got '{loss}'")
    
    # Check dataset description sufficiency
    if not methodology.dataset_description:
        gaps.append("dataset_description: missing entirely")
    else:
        ds = methodology.dataset_description
        
        # Dataset name
        ds_name = _extract_value(ds.name) if hasattr(ds, 'name') else None
        if not ds_name or len(ds_name) < 3:
            gaps.append(f"dataset_description.name: too vague -- got '{ds_name}'")
        elif ds_name.lower() in ["dataset", "data", "images", "unknown"]:
            gaps.append(f"dataset_description.name: generic placeholder -- got '{ds_name}'")
        
        # Input shape
        if not ds.input_shape or (isinstance(ds.input_shape, list) and len(ds.input_shape) == 0):
            gaps.append("dataset_description.input_shape: missing (required for model input layer)")
        
        # Number of classes (for classification)
        if hasattr(ds, 'num_classes'):
            nc = ds.num_classes
            if not nc or nc == 0:
                gaps.append("dataset_description.num_classes: missing or zero (required for output layer)")
            elif nc < 2 or nc > 100000:
                gaps.append(f"dataset_description.num_classes: unrealistic value -- got {nc}")
    
    # Check evaluation sufficiency
    if not methodology.evaluation:
        # Evaluation can be inferred for standard classification, so only warn
        pass
    else:
        # Metric name
        metric = _extract_value(methodology.evaluation.metric_name)
        if metric:
            metric_lower = metric.lower()
            if metric_lower in ["metric", "score", "unknown"]:
                gaps.append(f"evaluation.metric_name: generic placeholder -- got '{metric}'")
    
    return (len(gaps) == 0, gaps)


def format_readiness_report(methodology: MethodologySpec, gaps: list[str]) -> str:
    """Format a human-readable readiness report."""
    if not gaps:
        return "✅ CodeGen-ready: All fields have sufficient detail"
    
    lines = ["⚠️  CodeGen readiness gaps found:"]
    for i, gap in enumerate(gaps, 1):
        lines.append(f"  {i}. {gap}")
    
    return "\n".join(lines)
