import numpy as np
from datatrees import datatree, dtfield, Node


@datatree
class BoundingBox:
    """3D bounding box with min and max points."""

    min_point: np.ndarray = dtfield(
        default_factory=lambda: np.array([float("inf"), float("inf"), float("inf")])
    )
    max_point: np.ndarray = dtfield(
        default_factory=lambda: np.array([float("-inf"), float("-inf"), float("-inf")])
    )

    @property
    def size(self) -> np.ndarray:
        """Get the size of the bounding box as a 3D vector."""
        return self.max_point - self.min_point

    @property
    def center(self) -> np.ndarray:
        """Get the center of the bounding box."""
        # Ensure we always return a 3D vector even for an empty bounding box
        if np.all(np.isinf(self.min_point)) or np.all(np.isinf(self.max_point)):
            return np.array([0.0, 0.0, 0.0])
        return (self.max_point + self.min_point) / 2.0

    @property
    def diagonal(self) -> float:
        """Get the diagonal length of the bounding box."""
        if np.all(np.isinf(self.min_point)) or np.all(np.isinf(self.max_point)):
            return 1.0  # Return a default value for empty/invalid bounding boxes
        return np.linalg.norm(self.size)

    def union(self, other: "BoundingBox") -> "BoundingBox":
        """Compute the union of this bounding box with another."""
        # Handle the case where one of the bounding boxes is empty
        if np.all(np.isinf(self.min_point)):
            return other
        if np.all(np.isinf(other.min_point)):
            return self

        return BoundingBox(
            min_point=np.minimum(self.min_point, other.min_point),
            max_point=np.maximum(self.max_point, other.max_point),
        )

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the bounding box."""
        if np.all(np.isinf(self.min_point)) or np.all(np.isinf(self.max_point)):
            return False
        return np.all(point >= self.min_point) and np.all(point <= self.max_point)
