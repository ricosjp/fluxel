"""Three-component values in the public API.

``Int3`` is three integers, such as root counts ``(nx, ny, nz)``.
``Float3`` is three floats, such as a point ``(x, y, z)``.
``Bool3`` is three booleans, such as periodic flags ``(x, y, z)``.
"""

type Int3 = tuple[int, int, int]
type Float3 = tuple[float, float, float]
type Bool3 = tuple[bool, bool, bool]
