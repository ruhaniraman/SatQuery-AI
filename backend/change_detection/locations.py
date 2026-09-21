"""Plain-language positions in a scene (shared by change detection, scan comparison and reports)."""


def describe_location(cx: float, cy: float, width: float, height: float) -> str:
    """Plain-language position of a point in the scene, on a 3x3 grid."""
    col = min(2, int(3 * cx / width))
    row = min(2, int(3 * cy / height))
    if row == 1 and col == 1:
        return "centre"
    return f"{('upper', 'middle', 'lower')[row]}-{('left', 'centre', 'right')[col]}".replace("middle-", "")
