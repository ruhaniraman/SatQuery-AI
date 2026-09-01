from agent_manager.schemas import ClassificationResult

result = ClassificationResult(
    task="CHANGE_DETECTION",
    confidence=0.95
)

print(result)