//! Arc-length intersection of a constant-radius, constant-z path with world triangles.

const RELATIVE_TOLERANCE: f64 = 1e-9;

pub(super) struct ArcHit {
    pub distance: f64,
    pub point: [f64; 3],
    pub tangent: [f64; 3],
    pub normal: [f64; 3],
    pub anchor: usize,
}

/// Smallest forward arc hit from `theta` along `sign`.
///
/// `sign` is `+1` for increasing angle and `-1` for decreasing angle. `travel` is
/// the positive angle at which the path stops. A hit at the start has distance zero.
/// The tangent is `sign * e_θ` at the hit. The normal follows the triangle winding.
pub(super) fn first_hit(
    center: [f64; 3],
    radius: f64,
    theta: f64,
    travel: f64,
    sign: f64,
    triangles: &[[[f64; 3]; 3]],
) -> Option<ArcHit> {
    if !(radius.is_finite() && radius > 0.0 && travel.is_finite() && travel >= 0.0) {
        return None;
    }
    let mut best: Option<ArcHit> = None;
    for (anchor, triangle) in triangles.iter().copied().enumerate() {
        let Some(contact) = triangle_distance(center, radius, theta, travel, sign, triangle) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|current| contact.distance < current.distance)
        {
            let tangent = [-sign * contact.angle.sin(), sign * contact.angle.cos(), 0.0];
            best = Some(ArcHit {
                distance: contact.distance,
                point: contact.point,
                tangent,
                normal: contact.normal,
                anchor,
            });
        }
    }
    best
}

struct ArcContact {
    distance: f64,
    angle: f64,
    point: [f64; 3],
    normal: [f64; 3],
}

fn triangle_distance(
    center: [f64; 3],
    radius: f64,
    theta: f64,
    travel: f64,
    sign: f64,
    triangle: [[f64; 3]; 3],
) -> Option<ArcContact> {
    let normal = unit_normal(triangle)?;
    // Keep the first interior point whose forward arc stays inside the travel angle.
    let mut best = None;
    for psi in candidate_angles(center, radius, triangle) {
        let point = [
            center[0] + radius * psi.cos(),
            center[1] + radius * psi.sin(),
            center[2],
        ];
        if !contains_point(triangle, point) {
            continue;
        }
        let phi = forward_angle(theta, psi, sign);
        if phi <= travel {
            let distance = radius * phi;
            if best
                .as_ref()
                .is_none_or(|current: &ArcContact| distance < current.distance)
            {
                best = Some(ArcContact {
                    distance,
                    angle: psi,
                    point,
                    normal,
                });
            }
        }
    }
    best
}

fn unit_normal(triangle: [[f64; 3]; 3]) -> Option<[f64; 3]> {
    let normal = cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]));
    let length = hypot3(normal);
    if length == 0.0 {
        None
    } else {
        Some([normal[0] / length, normal[1] / length, normal[2] / length])
    }
}

fn candidate_angles(center: [f64; 3], radius: f64, triangle: [[f64; 3]; 3]) -> Vec<f64> {
    let edge_ab = sub(triangle[1], triangle[0]);
    let edge_ac = sub(triangle[2], triangle[0]);
    let normal = cross(edge_ab, edge_ac);
    let normal_length = hypot3(normal);
    if normal_length == 0.0 {
        return Vec::new();
    }
    // Plane equation A cos ψ + B sin ψ = C for the circle in that plane.
    let cosine_coeff = normal[0] * radius;
    let sine_coeff = normal[1] * radius;
    let rhs = dot(normal, triangle[0])
        - normal[0] * center[0]
        - normal[1] * center[1]
        - normal[2] * center[2];
    let amplitude = cosine_coeff.hypot(sine_coeff);
    let scale = normal_length * radius.max(1.0);
    // A horizontal circle in the triangle plane has no unique angle; use the edges.
    if amplitude <= RELATIVE_TOLERANCE * scale {
        if rhs.abs() > RELATIVE_TOLERANCE * scale {
            return Vec::new();
        }
        return coplanar_angles(center, radius, triangle);
    }
    if rhs.abs() > amplitude * (1.0 + 1e-12) {
        return Vec::new();
    }
    // Two candidate angles. The caller discards those outside the triangle.
    let alpha = sine_coeff.atan2(cosine_coeff);
    let beta = (rhs / amplitude).clamp(-1.0, 1.0).acos();
    vec![alpha + beta, alpha - beta]
}

fn coplanar_angles(center: [f64; 3], radius: f64, triangle: [[f64; 3]; 3]) -> Vec<f64> {
    let mut angles = Vec::new();
    for pair in [(0, 1), (1, 2), (2, 0)] {
        angles.extend(segment_angles(
            center,
            radius,
            triangle[pair.0],
            triangle[pair.1],
        ));
    }
    angles
}

fn segment_angles(center: [f64; 3], radius: f64, start: [f64; 3], end: [f64; 3]) -> Vec<f64> {
    // Circle-segment quadratic in the XY projection, then keep hits at this z.
    let edge = sub(end, start);
    let offset = [start[0] - center[0], start[1] - center[1]];
    let quadratic = edge[0] * edge[0] + edge[1] * edge[1];
    if quadratic == 0.0 {
        return Vec::new();
    }
    let linear = 2.0 * (offset[0] * edge[0] + offset[1] * edge[1]);
    let constant = offset[0] * offset[0] + offset[1] * offset[1] - radius * radius;
    let discriminant = linear * linear - 4.0 * quadratic * constant;
    if discriminant < 0.0 {
        return Vec::new();
    }
    let roots = if discriminant == 0.0 {
        vec![-linear / (2.0 * quadratic)]
    } else {
        let root = discriminant.sqrt();
        vec![
            (-linear - root) / (2.0 * quadratic),
            (-linear + root) / (2.0 * quadratic),
        ]
    };
    let z_tolerance = RELATIVE_TOLERANCE * radius.max(1.0);
    roots
        .into_iter()
        .filter(|step| (0.0..=1.0).contains(step))
        .filter_map(|step| {
            let point = [
                start[0] + step * edge[0],
                start[1] + step * edge[1],
                start[2] + step * edge[2],
            ];
            if (point[2] - center[2]).abs() > z_tolerance {
                return None;
            }
            Some((point[1] - center[1]).atan2(point[0] - center[0]))
        })
        .collect()
}

fn forward_angle(start: f64, angle: f64, sign: f64) -> f64 {
    if sign >= 0.0 {
        (angle - start).rem_euclid(std::f64::consts::TAU)
    } else {
        (start - angle).rem_euclid(std::f64::consts::TAU)
    }
}

fn contains_point(triangle: [[f64; 3]; 3], point: [f64; 3]) -> bool {
    // Barycentric coordinates. A near-zero area means the triangle is degenerate.
    let edge_ab = sub(triangle[1], triangle[0]);
    let edge_ac = sub(triangle[2], triangle[0]);
    let offset = sub(point, triangle[0]);
    let dot_ab_ab = dot(edge_ab, edge_ab);
    let dot_ab_ac = dot(edge_ab, edge_ac);
    let dot_ab_ap = dot(edge_ab, offset);
    let dot_ac_ac = dot(edge_ac, edge_ac);
    let dot_ac_ap = dot(edge_ac, offset);
    let determinant = dot_ab_ab * dot_ac_ac - dot_ab_ac * dot_ab_ac;
    if determinant.abs() <= RELATIVE_TOLERANCE * dot_ab_ab.max(dot_ac_ac).max(1.0) {
        return false;
    }
    let inverse = 1.0 / determinant;
    let u = (dot_ac_ac * dot_ab_ap - dot_ab_ac * dot_ac_ap) * inverse;
    let v = (dot_ab_ab * dot_ac_ap - dot_ab_ac * dot_ab_ap) * inverse;
    u >= -1e-8 && v >= -1e-8 && u + v <= 1.0 + 1e-8
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn hypot3(value: [f64; 3]) -> f64 {
    value[0].hypot(value[1]).hypot(value[2])
}
