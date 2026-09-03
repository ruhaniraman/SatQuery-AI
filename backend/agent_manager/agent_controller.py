from agent_manager.schemas import ClassificationResult, ImageInput,ExecutionTrace


# Dummy specialist functions
def single_image_vqa():
    print("Running Single Image VQA")
    return {"status": "success"}


def change_detection():
    print("Running Change Detection")
    return {"status": "success"}


def grounding():
    print("Running Grounding")
    return {"status": "success"}


def optical_sar_fusion():
    print("Running Optical-SAR Fusion")
    return {"status": "success"}
# Tool Registry
TOOL_REGISTRY = {
    "SINGLE_IMAGE_VQA": single_image_vqa,
    "CHANGE_DETECTION": change_detection,
    "GROUNDING": grounding,
    "OPTICAL_SAR_FUSION": optical_sar_fusion
}

# Query classification
def classify_query(query):

    query = query.lower()

    if "change" in query or "changed" in query:
        task = "CHANGE_DETECTION"
        confidence = 0.95

    elif "where" in query or "locate" in query:
        task = "GROUNDING"
        confidence = 0.90

    elif "sar" in query or "optical" in query:
        task = "OPTICAL_SAR_FUSION"
        confidence = 0.90

    else:
        task = "SINGLE_IMAGE_VQA"
        confidence = 0.80

    return ClassificationResult(
        task=task,
        confidence=confidence
    )


# Input validation
def validate_inputs(task, images):

    if task in [
        "SINGLE_IMAGE_VQA",
        "GROUNDING"
    ]:
        if len(images) < 1:
            return False, "This task requires at least one image."

    elif task == "CHANGE_DETECTION":

        if len(images) < 2:
            return False, "Change detection requires two images."

        if not images[0].date or not images[1].date:
            return False, "Both images must have dates."

        if images[0].date == images[1].date:
            return False, "The two images must have different dates."

    elif task == "OPTICAL_SAR_FUSION":

        if len(images) < 2:
            return False, "Optical-SAR fusion requires two images."

        modalities = {
            image.modality.lower()
            for image in images
        }

        has_optical = (
            "optical" in modalities
            or "multispectral" in modalities
        )

        has_sar = "sar" in modalities

        if not has_optical or not has_sar:
            return False, (
                "You need one optical/multispectral image "
                "and one SAR image."
            )

    return True, None


# Routing
def route_query(query, images):

    classification = classify_query(query)
    task = classification.task

    print("Query:", query)
    print("Selected Task:", task)
    print("Confidence:", classification.confidence)

    # Validate inputs
    valid, error = validate_inputs(task, images)

    if not valid:

        print("Validation Failed:", error)

        trace = ExecutionTrace(
            query=query,
            task=task,
            confidence=classification.confidence,
            validation_status="failed",
            execution_status="not_executed"
        )

        print("Execution Trace:", trace)

        return {
            "status": "error",
            "message": error,
            "trace": trace.model_dump()
        }

    print("Validation Passed")

    # Find the correct specialist using the registry
    tool = TOOL_REGISTRY.get(task.value)

    if tool is None:
        return {
            "status": "error",
            "message": "No specialist found for this task."
        }

    # Run the selected specialist
    result = tool()

    # Create execution trace
    trace = ExecutionTrace(
        query=query,
        task=task,
        confidence=classification.confidence,
        validation_status="passed",
        execution_status=result["status"]
    )

    print("Execution Trace:", trace)

    return {
        **result,
        "trace": trace.model_dump()
    }

# Test validation
if __name__ == "__main__":

    test_cases = [

        # 1. Change detection with only 1 image
        {
            "name": "Change Detection - 1 Image",
            "query": "What changed?",
            "images": [
                ImageInput(
                    path="image1.tif",
                    modality="optical",
                    date="2025-01-10"
                )
            ]
        },

        # 2. Change detection with same dates
        {
            "name": "Change Detection - Same Dates",
            "query": "What changed?",
            "images": [
                ImageInput(
                    path="image1.tif",
                    modality="optical",
                    date="2025-01-10"
                ),
                ImageInput(
                    path="image2.tif",
                    modality="optical",
                    date="2025-01-10"
                )
            ]
        },

        # 3. Change detection with correct inputs
        {
            "name": "Change Detection - Correct",
            "query": "What changed?",
            "images": [
                ImageInput(
                    path="image1.tif",
                    modality="optical",
                    date="2025-01-10"
                ),
                ImageInput(
                    path="image2.tif",
                    modality="optical",
                    date="2025-02-10"
                )
            ]
        },

        # 4. Optical + SAR with correct inputs
        {
            "name": "Optical + SAR - Correct",
            "query": "Analyze the optical and SAR images.",
            "images": [
                ImageInput(
                    path="optical.tif",
                    modality="optical",
                    date="2025-01-10"
                ),
                ImageInput(
                    path="sar.tif",
                    modality="sar",
                    date="2025-01-10"
                )
            ]
        },

        # 5. Optical + Optical - Incorrect
        {
            "name": "Optical + SAR - Missing SAR",
            "query": "Analyze the optical and SAR images.",
            "images": [
                ImageInput(
                    path="optical1.tif",
                    modality="optical",
                    date="2025-01-10"
                ),
                ImageInput(
                    path="optical2.tif",
                    modality="optical",
                    date="2025-02-10"
                )
            ]
        }
    ]

    for test in test_cases:

        print("\nTest:", test["name"])

        result = route_query(
            test["query"],
            test["images"]
        )

        print("Result:", result)
        print("-------------------------")