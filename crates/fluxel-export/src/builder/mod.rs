//! Builders that extract interior faces and boundary topology in bulk.

mod axis_projected;
mod ghost_cell;

pub use axis_projected::build_axis_projected_mesh;
pub use ghost_cell::build_ghost_cell_mesh;
