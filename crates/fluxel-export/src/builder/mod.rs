//! Builders that extract interior faces and boundary topology in bulk.

mod axis_projected;
mod ghost_cell;

pub use axis_projected::{
    build_axis_projected_mesh, fill_ap_ibm_face_data, has_under_refined_intersect_cells,
    rebuild_axis_projected_ib,
};
pub use ghost_cell::build_ghost_cell_mesh;
