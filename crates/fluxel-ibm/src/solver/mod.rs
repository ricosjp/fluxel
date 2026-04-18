//! Solver routines for intersection and inside/outside classification.

mod apibm;
mod classify;
mod flood;
mod gcibm;
mod locate;

pub use apibm::{resolve_apibm_face, ApibmIntersection};
pub use classify::mark_intersecting_cells;
pub use flood::flood_fill_inside_outside;
pub use gcibm::compute_ghost_cell_geometry;
pub use locate::get_global_id_from_phys;
