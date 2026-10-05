//! Exact annular-sector intersection with one world triangle.
//!
//! Cartesian cells use an axis-aligned box test in the immersed-boundary classifier.
//! A cylindrical cell uses this convex wedge test after that same box filter.

use super::domain::{CylindricalGeometry, ParameterInterval};

impl CylindricalGeometry {
    /// True when a world triangle meets the closed annular sector.
    ///
    /// The sector angle must be at most `π/2`, so the wedge is convex. Contact on
    /// the boundary counts as an intersection.
    pub fn intersects_triangle(
        &self,
        interval: ParameterInterval,
        triangle: [[f64; 3]; 3],
    ) -> bool {
        let (aabb_center, aabb_size) = self.world_aabb(interval);
        if !aabb_overlaps_triangle(aabb_center, aabb_size, triangle) {
            return false;
        }
        // Clip to the convex wedge: z slab, then the two angular half-planes.
        let mut polygon = triangle.to_vec();
        let origin = self.origin();
        let z0 = origin[2] + interval.min[2];
        let z1 = origin[2] + interval.max[2];
        polygon = clip_halfspace(&polygon, |point| point[2] - z0);
        polygon = clip_halfspace(&polygon, |point| z1 - point[2]);
        let start = [interval.min[1].cos(), interval.min[1].sin()];
        let end = [interval.max[1].cos(), interval.max[1].sin()];
        polygon = clip_halfspace(&polygon, |point| {
            cross_z(start, [point[0] - origin[0], point[1] - origin[1]])
        });
        polygon = clip_halfspace(&polygon, |point| {
            -cross_z(end, [point[0] - origin[0], point[1] - origin[1]])
        });
        // An empty clip is a miss. A remaining point or edge is still contact.
        if polygon.is_empty() {
            return false;
        }
        let (min_radius, max_radius) = radius_range(origin, &polygon);
        min_radius <= interval.max[0] && max_radius >= interval.min[0]
    }
}

fn cross_z(direction: [f64; 2], point: [f64; 2]) -> f64 {
    direction[0] * point[1] - direction[1] * point[0]
}

fn clip_halfspace(polygon: &[[f64; 3]], inside: impl Fn([f64; 3]) -> f64) -> Vec<[f64; 3]> {
    if polygon.is_empty() {
        return Vec::new();
    }
    // Sutherland–Hodgman: emit the crossing, then keep vertices on the inside or boundary.
    let mut output = Vec::new();
    let mut previous = polygon[polygon.len() - 1];
    let mut previous_side = inside(previous);
    for &current in polygon {
        let current_side = inside(current);
        let crosses = previous_side < 0.0 && current_side >= 0.0
            || previous_side >= 0.0 && current_side < 0.0;
        if crosses {
            let step = previous_side / (previous_side - current_side);
            output.push([
                previous[0] + step * (current[0] - previous[0]),
                previous[1] + step * (current[1] - previous[1]),
                previous[2] + step * (current[2] - previous[2]),
            ]);
        }
        if current_side >= 0.0 {
            output.push(current);
        }
        previous = current;
        previous_side = current_side;
    }
    output
}

fn aabb_overlaps_triangle(center: [f64; 3], size: [f64; 3], triangle: [[f64; 3]; 3]) -> bool {
    let half = [0.5 * size[0], 0.5 * size[1], 0.5 * size[2]];
    let local = triangle.map(|point| {
        [
            point[0] - center[0],
            point[1] - center[1],
            point[2] - center[2],
        ]
    });
    let edges = [
        sub(local[1], local[0]),
        sub(local[2], local[1]),
        sub(local[0], local[2]),
    ];
    // Separating-axis test: box faces, then each edge crossed with a box axis.
    for axis in 0..3 {
        let min = local
            .iter()
            .map(|point| point[axis])
            .fold(f64::MAX, f64::min);
        let max = local
            .iter()
            .map(|point| point[axis])
            .fold(f64::MIN, f64::max);
        if max < -half[axis] || min > half[axis] {
            return false;
        }
    }
    for edge in edges {
        for axis in 0..3 {
            let normal = match axis {
                0 => [0.0, -edge[2], edge[1]],
                1 => [edge[2], 0.0, -edge[0]],
                _ => [-edge[1], edge[0], 0.0],
            };
            if separated(local, half, normal) {
                return false;
            }
        }
    }
    // Last axis is the triangle normal.
    !separated(local, half, cross(edges[0], edges[1]))
}

fn separated(points: [[f64; 3]; 3], half: [f64; 3], axis: [f64; 3]) -> bool {
    let length = axis[0].hypot(axis[1]).hypot(axis[2]);
    if length == 0.0 {
        return false;
    }
    let projected: Vec<f64> = points
        .iter()
        .map(|point| point[0] * axis[0] + point[1] * axis[1] + point[2] * axis[2])
        .collect();
    let min = projected.iter().copied().fold(f64::MAX, f64::min);
    let max = projected.iter().copied().fold(f64::MIN, f64::max);
    let radius = half[0] * axis[0].abs() + half[1] * axis[1].abs() + half[2] * axis[2].abs();
    max < -radius || min > radius
}

fn radius_range(origin: [f64; 3], polygon: &[[f64; 3]]) -> (f64, f64) {
    let mut min_radius = f64::MAX;
    let mut max_radius = 0.0_f64;
    for window in polygon.windows(2) {
        let start = [window[0][0], window[0][1]];
        let end = [window[1][0], window[1][1]];
        min_radius = min_radius.min(segment_min_radius(origin, start, end));
        max_radius = max_radius
            .max(point_radius(origin, start))
            .max(point_radius(origin, end));
    }
    let first = [polygon[0][0], polygon[0][1]];
    let last = [polygon[polygon.len() - 1][0], polygon[polygon.len() - 1][1]];
    min_radius = min_radius.min(segment_min_radius(origin, last, first));
    max_radius = max_radius
        .max(point_radius(origin, first))
        .max(point_radius(origin, last));
    (min_radius, max_radius)
}

fn point_radius(origin: [f64; 3], point: [f64; 2]) -> f64 {
    (point[0] - origin[0]).hypot(point[1] - origin[1])
}

fn segment_min_radius(origin: [f64; 3], start: [f64; 2], end: [f64; 2]) -> f64 {
    let edge = [end[0] - start[0], end[1] - start[1]];
    let length2 = edge[0] * edge[0] + edge[1] * edge[1];
    if length2 == 0.0 {
        return point_radius(origin, start);
    }
    let toward = [origin[0] - start[0], origin[1] - start[1]];
    let step = ((toward[0] * edge[0] + toward[1] * edge[1]) / length2).clamp(0.0, 1.0);
    point_radius(
        origin,
        [start[0] + step * edge[0], start[1] + step * edge[1]],
    )
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use fluxel_sfc::LogicalCell;

    #[test]
    fn triangle_intersection_uses_the_annular_wedge() {
        let domain = CylindricalGeometry::full_turn([0.0; 3], 1.0, 3.0, 0.0, 0.0, 1.0).unwrap();
        let interval = domain
            .cell_interval(&LogicalCell::root(0), [2, 4, 1])
            .unwrap();
        let through = [[2.0, 0.2, 0.2], [2.0, 0.2, 0.8], [0.2, 2.0, 0.5]];
        assert!(domain.intersects_triangle(interval, through));
        let outside = [[4.0, 0.2, 0.2], [4.0, 0.2, 0.8], [3.5, 2.0, 0.5]];
        assert!(!domain.intersects_triangle(interval, outside));
        let point = [[2.0, 0.0, 0.5], [3.0, -1.0, 0.5], [3.0, -1.0, 2.0]];
        assert!(domain.intersects_triangle(interval, point));
    }
}
