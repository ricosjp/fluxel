//! Ghost-cell geometry and trilinear stencil construction.

use super::classify::check_inside_parity_robust;
use super::locate::get_global_id_from_phys;
use crate::mesh::IBMMesh;
use crate::types::{CellType, GhostCellData};
use fluxel_core::{Direction, Forest};
use fluxel_geometry::Geometry;
use nalgebra::{Matrix4, Vector4};
use parry3d_f64::math::{Pose, Vector};
use parry3d_f64::query::PointQuery;
use rayon::prelude::*;

#[inline]
fn fluid_cell_from_phys(
    forest: &Forest,
    geom: &Geometry,
    gc_is_fluid: &[bool],
    px: f64,
    py: f64,
    pz: f64,
) -> Option<usize> {
    get_global_id_from_phys(forest, geom, [px, py, pz]).filter(|&gid| gc_is_fluid[gid])
}

/// One computational ghost cell: boundary data and trilinear stencil row.
struct GhostStencilRow {
    cell_id: usize,
    bnd_anchor_id: usize,
    bnd_patch_id: usize,
    bnd_intercept: [f64; 3],
    image_point: [f64; 3],
    stencil_indices: [usize; 8],
    stencil_weights: [f64; 8],
}

fn try_build_ghost_stencil_row(
    global_id: usize,
    forest: &Forest,
    geom: &Geometry,
    mesh: &IBMMesh,
    gc_is_fluid: &[bool],
) -> Option<GhostStencilRow> {
    let identity = Pose::identity();
    let logical = forest.keys()[global_id].to_logical();

    if gc_is_fluid[global_id] {
        return None;
    }

    let touches_fluid = Direction::ALL.iter().any(|&dir| {
        forest
            .face_neighbour_global_ids(global_id, dir)
            .iter()
            .any(|&nbr_id| gc_is_fluid[nbr_id])
    });

    if !touches_fluid {
        return None;
    }

    let (center_arr, size_arr) = geom.cell_bounds(&logical);
    let center = Vector::new(center_arr[0], center_arr[1], center_arr[2]);

    let (proj, feature) = mesh.bvh.project_point_and_get_feature(&identity, center);
    let closest_point = proj.point;
    let anchor_id = feature.unwrap_face() as usize;
    let patch_id = mesh.anchor_to_patch_id[anchor_id];

    let image_point = Vector::new(
        2.0 * closest_point.x - center.x,
        2.0 * closest_point.y - center.y,
        2.0 * closest_point.z - center.z,
    );

    let mut stencil_indices = [0usize; 8];
    let mut stencil_weights = [0.0f64; 8];

    let eps_x = size_arr[0] * 0.1;
    let eps_y = size_arr[1] * 0.1;
    let eps_z = size_arr[2] * 0.1;

    let offsets = [
        [-eps_x, -eps_y, -eps_z],
        [eps_x, -eps_y, -eps_z],
        [-eps_x, eps_y, -eps_z],
        [eps_x, eps_y, -eps_z],
        [-eps_x, -eps_y, eps_z],
        [eps_x, -eps_y, eps_z],
        [-eps_x, eps_y, eps_z],
        [eps_x, eps_y, eps_z],
    ];

    let mut unique_fluid_cells = Vec::new();
    for offset in &offsets {
        if let Some(idx) = fluid_cell_from_phys(
            forest,
            geom,
            gc_is_fluid,
            image_point.x + offset[0],
            image_point.y + offset[1],
            image_point.z + offset[2],
        ) {
            if !unique_fluid_cells.contains(&idx) {
                unique_fluid_cells.push(idx);
            }
        }
    }

    let num_points = unique_fluid_cells.len().min(8);

    let mut m_mat = Matrix4::zeros();
    let mut a_row = [Vector4::zeros(); 8];
    let mut idw_w = [0.0; 8];

    let l_ref = size_arr[0].max(1e-12);

    for (k, &fluid_global_id) in unique_fluid_cells.iter().take(num_points).enumerate() {
        let (c_arr, _) = geom.cell_bounds(&forest.keys()[fluid_global_id].to_logical());

        let dx = (c_arr[0] - image_point.x) / l_ref;
        let dy = (c_arr[1] - image_point.y) / l_ref;
        let dz = (c_arr[2] - image_point.z) / l_ref;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        let w = if dist < 1e-9 { 1e9 } else { 1.0 / dist };
        idw_w[k] = w;

        let a_k = Vector4::new(1.0, dx, dy, dz);
        a_row[k] = a_k;

        m_mat += a_k * a_k.transpose() * w;
    }

    let mut use_idw_fallback = true;

    if num_points >= 4 {
        if let Some(inv_m) = m_mat.try_inverse() {
            use_idw_fallback = false;
            let mut sum_w = 0.0;

            let row_0 = Vector4::new(inv_m[(0, 0)], inv_m[(0, 1)], inv_m[(0, 2)], inv_m[(0, 3)]);

            (0..num_points).for_each(|k| {
                let final_w = row_0.dot(&a_row[k]);

                let weight = idw_w[k] * final_w;
                stencil_weights[k] = weight;
                stencil_indices[k] = unique_fluid_cells[k];
                sum_w += weight;
            });

            if sum_w.abs() > 1e-12 {
                stencil_weights
                    .iter_mut()
                    .take(num_points)
                    .for_each(|w| *w /= sum_w);
            } else {
                use_idw_fallback = true;
            }
        }
    }

    if use_idw_fallback {
        let mut sum_w = 0.0;
        (0..num_points).for_each(|k| {
            stencil_weights[k] = idw_w[k];
            stencil_indices[k] = unique_fluid_cells[k];
            sum_w += idw_w[k];
        });
        if sum_w > 0.0 {
            stencil_weights
                .iter_mut()
                .take(num_points)
                .for_each(|w| *w /= sum_w);
        } else {
            stencil_indices[0] = global_id;
            stencil_weights[0] = 1.0;
        }
    }

    Some(GhostStencilRow {
        cell_id: global_id,
        bnd_anchor_id: anchor_id,
        bnd_patch_id: patch_id,
        bnd_intercept: [closest_point.x, closest_point.y, closest_point.z],
        image_point: [image_point.x, image_point.y, image_point.z],
        stencil_indices,
        stencil_weights,
    })
}

/// Steps 4 and 5: identify ghost cells and build image-point interpolation stencils.
///
/// Extracts `Solid`/`Intersect` cells that share a face with `Fluid` cells as computational
/// ghost cells, then constructs mirrored image points across the boundary surface.
/// It also locates surrounding fluid cells and stores interpolation stencil indices/weights.
pub fn compute_ghost_cell_geometry(
    forest: &Forest,
    geom: &Geometry,
    mesh: &IBMMesh,
    cell_types: &[CellType],
    fluid_seed_point: [f64; 3],
) -> GhostCellData {
    let total_cells = forest.num_cells();
    let mut gc_is_fluid = vec![false; total_cells];

    gc_is_fluid
        .par_iter_mut()
        .zip(cell_types.par_iter())
        .for_each(|(slot, ct)| {
            if *ct == CellType::Fluid {
                *slot = true;
            }
        });

    let seed_pt = Vector::new(
        fluid_seed_point[0],
        fluid_seed_point[1],
        fluid_seed_point[2],
    );
    let fluid_seed_is_inside = check_inside_parity_robust(&seed_pt, &mesh.bvh);

    gc_is_fluid
        .par_iter_mut()
        .zip(forest.keys().par_iter())
        .zip(cell_types.par_iter())
        .for_each(|((slot, key), ct)| {
            if *ct != CellType::Intersect {
                return;
            }
            let logical = key.to_logical();
            let (c_arr, _) = geom.cell_bounds(&logical);
            let center = Vector::new(c_arr[0], c_arr[1], c_arr[2]);
            let is_inside = check_inside_parity_robust(&center, &mesh.bvh);
            *slot = is_inside == fluid_seed_is_inside;
        });

    let rows: Vec<Option<GhostStencilRow>> = (0..total_cells)
        .into_par_iter()
        .map(|gid| try_build_ghost_stencil_row(gid, forest, geom, mesh, &gc_is_fluid))
        .collect();

    let mut gc_cell_ids = Vec::new();
    let mut gc_bnd_anchor_ids = Vec::new();
    let mut gc_bnd_patch_ids = Vec::new();
    let mut gc_bnd_intercepts = Vec::new();
    let mut gc_image_points = Vec::new();
    let mut gc_interp_stencil_indices = Vec::new();
    let mut gc_interp_stencil_weights = Vec::new();

    for row in rows.into_iter().flatten() {
        gc_cell_ids.push(row.cell_id);
        gc_bnd_anchor_ids.push(row.bnd_anchor_id);
        gc_bnd_patch_ids.push(row.bnd_patch_id);
        gc_bnd_intercepts.push(row.bnd_intercept);
        gc_image_points.push(row.image_point);
        gc_interp_stencil_indices.push(row.stencil_indices);
        gc_interp_stencil_weights.push(row.stencil_weights);
    }

    GhostCellData {
        gc_is_fluid,
        gc_cell_ids,
        gc_bnd_anchor_ids,
        gc_bnd_patch_ids,
        gc_bnd_intercepts,
        gc_image_points,
        gc_interp_stencil_indices,
        gc_interp_stencil_weights,
    }
}
