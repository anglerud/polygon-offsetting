pub mod draw_svg;
pub mod polygon_offsetting;
pub use crate::polygon_offsetting::Offset;
pub use crate::polygon_offsetting::Polygon;

// Define our errors
#[derive(Debug)]
pub enum OffsetError {
    InvalidTolerance,
    NoValidRegions,
    CollapsedPolygon,
    UnclosedPolygon,
    RegionSortingFailed,
    SinglePointRegion,
    InvalidResult,
}

impl std::fmt::Display for OffsetError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            OffsetError::InvalidTolerance => {
                write!(f, "The tolerance can't be below or equal to 0")
            }
            OffsetError::NoValidRegions => {
                write!(f, "Offset operation resulted in no valid regions")
            }
            OffsetError::CollapsedPolygon => write!(
                f,
                "Offset operation resulted in no valid regions (collapsed polygon)"
            ),
            OffsetError::UnclosedPolygon => write!(f, "Unclosed polygon"),
            OffsetError::RegionSortingFailed => write!(f, "No valid regions found after sorting"),
            OffsetError::SinglePointRegion => write!(f, "Offset resulted in a single point region"),
            OffsetError::InvalidResult => write!(f, "Offset produced an invalid result"),
        }
    }
}

impl std::error::Error for OffsetError {}
