//! Python-independent mesh generation, refinement policy and transactional sessions.
mod build;
mod config;
mod error;
#[cfg(feature = "mesh-io")]
pub mod io;
mod manual;
mod refinement;
mod result;
mod session;
pub use build::{build_apibm, build_gcibm};
pub use config::{BuildLimits, MeshBuildConfig, ParameterRegion, RefinementPlan, RefinementRegion};
pub use error::BuildError;
pub use manual::ManualGrid;
pub use result::{ApibmMesh, BuildOutput, BuildReport, GcibmMesh, MeshSnapshot};
pub use session::{ApibmSession, PoseUpdate, PreparedUpdate, RemeshRequest};
