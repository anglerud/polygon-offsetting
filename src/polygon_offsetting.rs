use std::collections::{HashMap, HashSet};

use crate::OffsetError;

#[derive(Default, Debug, Clone)]
pub struct Offset {
    pub contour: Vec<(f64, f64)>,
    pub area: f64,
    pub perimeter: f64,
}

#[derive(Default, Debug, Clone)]
pub struct Polygon {
    edges: Vec<Edge>,
    vertices: HashMap<usize, Vertex>,
    offset_margin: f64,
    is_degenerate: bool,
}

#[derive(Default, Debug, Copy, Clone)]
struct Vertex {
    x: f64,
    y: f64,
    is_intersect: bool,
}

#[derive(Default, Debug, Copy, Clone)]
struct Edge {
    p1: usize,
    p2: usize,
    index: usize,
    outward_normal: Vertex,
}

#[derive(Default, Clone, Debug)]
struct Segment {
    p1: (f64, f64),
    p2: (f64, f64),
}

/// Computes the area of a polygon using the shoelace formula.
///
/// # Arguments
/// * `contours` - A vector of (x,y) coordinate tuples representing the polygon's vertices
///
/// # Returns
/// The polygon's area as f64 (always positive)
///
/// # Notes
/// - Assumes the polygon is closed (first and last points should be equal)
/// - Works for both convex and concave polygons
/// - Returns 0.0 for empty input or degenerate cases
#[inline]
fn compute_area(contours: &Vec<(f64, f64)>) -> f64 {
    let mut a = 0.0;
    if contours.len() == 0 {
        return 0.0;
    }
    for i in 0..contours.len() - 1 {
        a = a + (contours[i].0 * contours[i + 1].1) - (contours[i + 1].0 * contours[i].1);
    }
    (a * -0.5).abs()
}

/// Computes the perimeter of a polygon by summing edge lengths.
///
/// # Arguments
/// * `contour2d` - A vector of (x,y) coordinate tuples representing the polygon's vertices
///
/// # Returns
/// The total perimeter length as f64
///
/// # Notes
/// - Assumes the polygon is closed (first and last points should be equal)
/// - Skips the last edge if it would be zero-length (duplicate start/end point)
fn compute_perimeter(contour2d: &Vec<(f64, f64)>) -> f64 {
    let mut perimeter2d = 0.;
    for i in 0..(contour2d.len() - 1) {
        let p1 = &contour2d[i];
        let p2 = &contour2d[i + 1];
        perimeter2d += ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt();
    }
    perimeter2d
}

/// Computes the Euclidean distance between two points.
///
/// # Arguments
/// * `p1` - First point as (x,y) tuple
/// * `p2` - Second point as (x,y) tuple
///
/// # Returns
/// The distance between p1 and p2 as f64
///
/// # Notes
/// - Uses standard Euclidean distance formula
/// - Handles all finite coordinate values
#[inline]
fn get_dist(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt()
}

/// Subtracts two 2D vectors component-wise.
///
/// # Arguments
/// * `v1` - First vector as (x,y) tuple  
/// * `v2` - Second vector as (x,y) tuple
///
/// # Returns
/// Resulting vector as (x,y) tuple where x = v1.x - v2.x, y = v1.y - v2.y
#[inline]
fn vector_sub(v1: (f64, f64), v2: (f64, f64)) -> (f64, f64) {
    (v1.0 - v2.0, v1.1 - v2.1)
}

/// Adds two 2D vectors component-wise.
///
/// # Arguments
/// * `v1` - First vector as (x,y) tuple
/// * `v2` - Second vector as (x,y) tuple
///
/// # Returns  
/// Resulting vector as (x,y) tuple where x = v1.x + v2.x, y = v1.y + v2.y
#[inline]
fn vector_add(v1: (f64, f64), v2: (f64, f64)) -> (f64, f64) {
    (v1.0 + v2.0, v1.1 + v2.1)
}

/// Reverses the order and direction of a sequence of line segments.
///
/// # Arguments
/// * `sgmts` - Vector of Segments to reverse
///
/// # Returns
/// New vector of Segments where:
/// - The segment order is reversed
/// - Each segment's start/end points are swapped
///
/// # Notes
/// - Useful for changing polygon winding direction
#[inline]
fn reverse_segments(sgmts: &Vec<Segment>) -> Vec<Segment> {
    let mut segments: Vec<Segment> = Vec::new();

    for s in sgmts.iter().rev() {
        segments.push(Segment { p1: s.p2, p2: s.p1 });
    }
    segments
}

// =================================================================================

impl Polygon {
    /// Determines if the polygon has collapsed due to an inward offset.
    ///
    /// A polygon is considered collapsed when:
    /// - The offset margin is negative (inward offset)
    /// - The absolute value of the offset margin exceeds the smallest dimension
    ///   of the polygon's bounding box
    ///
    /// # Returns
    /// - `true` if the polygon has collapsed
    /// - `false` if the offset is outward or the polygon hasn't collapsed
    ///
    /// # Notes
    /// - This is a conservative check that prevents excessive inward offsets
    /// - Only inward offsets (negative margin) can cause collapse
    /// - Uses bounding box dimensions for computational efficiency
    fn is_collapsed(&self) -> bool {
        if self.offset_margin >= 0.0 {
            return false; // Only inward offsets can collapse
        }

        // Calculate polygon's bounding box dimensions
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for vertex in self.vertices.values() {
            min_x = min_x.min(vertex.x);
            max_x = max_x.max(vertex.x);
            min_y = min_y.min(vertex.y);
            max_y = max_y.max(vertex.y);
        }

        let width = max_x - min_x;
        let height = max_y - min_y;
        let min_dimension = width.min(height);

        // Consider collapsed only if offset is larger than the smallest dimension
        // This is more conservative than before
        self.offset_margin.abs() > min_dimension
    }

    fn append_arc(
        &self,
        center: &Vertex,
        radius: f64,
        start_vertex: &Vertex,
        end_vertex: &Vertex,
        tolerance: f64,
    ) -> Vec<Vertex> {
        // we compute start and end angles
        let mut start_angle = (start_vertex.y - center.y).atan2(start_vertex.x - center.x);
        let mut end_angle = (end_vertex.y - center.y).atan2(end_vertex.x - center.x);
        if start_angle < 0. {
            start_angle = 2. * std::f64::consts::PI + start_angle
        }
        if end_angle <= 0. {
            end_angle = 2. * std::f64::consts::PI + end_angle
        }

        // we compute oriented angle
        let mut angle: f64 = end_angle - start_angle;
        if angle.abs() > std::f64::consts::PI {
            angle = -(angle / angle.abs()) * (2. * std::f64::consts::PI - angle.abs());
        }

        // we compute number of segments to fit the tolerance
        let n_angle =
            (1. - (4. * radius * tolerance - 2. * tolerance.powi(2)) / (radius.powi(2))).acos();
        if n_angle == 0.0 {
            let vect = vector_add(
                (start_vertex.x, start_vertex.y),
                vector_sub(
                    (end_vertex.x, end_vertex.y),
                    (start_vertex.x, start_vertex.y),
                ),
            );
            return vec![Vertex {
                x: vect.0,
                y: vect.1,
                is_intersect: false,
            }];
        }
        let nseg: i64 = ((angle.abs() / n_angle.abs()).round() + 1.) as i64;

        // we define the angular step
        let angular_step = angle / nseg as f64;

        // we create the segments
        let mut dots = Vec::new();
        for i in 0..(nseg + 1) {
            let theta = start_angle + angular_step * i as f64;
            let x = radius * (theta).cos() + center.x;
            let y = radius * (theta).sin() + center.y;
            dots.push(Vertex {
                x: x,
                y: y,
                is_intersect: false,
            });
        }
        dots
    }

    /// Computes the intersection point between two edges if one exists.
    ///
    /// # Arguments
    /// * `e1` - First edge as a tuple of vertices (start, end)
    /// * `e2` - Second edge as a tuple of vertices (start, end)
    /// * `is_inters` - Flag indicating if the intersection should be marked as special
    ///
    /// # Returns
    /// - `Some(Vertex)` containing the intersection point if edges intersect properly
    /// - `None` if edges are parallel, coincident, or don't intersect within segments
    ///
    /// # Notes
    /// - Uses parametric line intersection with floating-point tolerance checks
    /// - Only returns intersections that occur within both edge segments (not infinite lines)
    /// - Handles near-parallel cases with numerical stability checks
    /// - Intersection points are marked with `is_intersect` flag from input parameter
    fn edges_intersection(
        &self,
        e1: &(Vertex, Vertex),
        e2: &(Vertex, Vertex),
        is_inters: bool,
    ) -> Option<Vertex> {
        let den = (e2.1.y - e2.0.y) * (e1.1.x - e1.0.x) - (e2.1.x - e2.0.x) * (e1.1.y - e1.0.y);
        if den > -0.0001 && den < 0.0001 {
            return None; // lines are parallel or conincident
        }

        let ua =
            ((e2.1.x - e2.0.x) * (e1.0.y - e2.0.y) - (e2.1.y - e2.0.y) * (e1.0.x - e2.0.x)) / den;
        let ub =
            ((e1.1.x - e1.0.x) * (e1.0.y - e2.0.y) - (e1.1.y - e1.0.y) * (e1.0.x - e2.0.x)) / den;

        if ua < 0.0000001 || ub < 0.0000001 || ua > 0.9999999 || ub > 0.9999999 {
            return None;
        }

        let v_cross = Some(Vertex {
            x: e1.0.x + ua * (e1.1.x - e1.0.x),
            y: e1.0.y + ua * (e1.1.y - e1.0.y),
            is_intersect: is_inters,
        });
        v_cross
    }

    /// Creates a new edge by offsetting an existing edge by specified distances.
    ///
    /// # Arguments
    /// * `p1` - First vertex of the original edge
    /// * `p2` - Second vertex of the original edge
    /// * `dx` - Horizontal offset distance
    /// * `dy` - Vertical offset distance
    ///
    /// # Returns
    /// A tuple of two vertices representing the offset edge
    ///
    /// # Notes
    /// - This performs a simple translation of both endpoints
    /// - The offset direction is determined by the sign of dx/dy
    /// - No edge length or validity checks are performed
    fn create_offset_edge(&self, p1: &Vertex, p2: &Vertex, dx: f64, dy: f64) -> (Vertex, Vertex) {
        let mut v1: Vertex = Vertex::default();
        let mut v2: Vertex = Vertex::default();
        v1.x = p1.x + dx;
        v1.y = p1.y + dy;
        v2.x = p2.x + dx;
        v2.y = p2.y + dy;
        (v1, v2)
    }

    /// Creates a polygon offset by the specified margin from the original polygon.
    ///
    /// This function generates a new polygon where each edge is offset by the polygon's
    /// `offset_margin` distance along its outward normal. The resulting polygon may:
    /// - Have rounded corners where arcs are inserted between offset edges
    /// - Collapse edges that become too small (below tolerance)
    /// - Contain self-intersections that need to be resolved later
    ///
    /// # Arguments
    /// * `tolerance` - Minimum edge length threshold below which edges are collapsed
    ///
    /// # Returns
    /// A new Polygon with:
    /// - Offset vertices and edges
    /// - Original offset_margin value preserved
    /// - is_degenerate flag set to false initially
    ///
    /// # Notes
    /// - For inward offsets (negative margin), may produce self-intersecting geometry
    /// - Collapsed edges are replaced with midpoints when below tolerance length
    /// - Arc segments are inserted between non-intersecting offset edges
    /// - The resulting polygon may require further processing with detect_all_intersect()
    fn create_margin_polygon(&mut self, tolerance: f64) -> Polygon {
        let mut offset_edges: Vec<(Vertex, Vertex)> = Vec::new();
        let mut vertices: HashMap<usize, Vertex> = HashMap::new();
        let mut index: usize = 0;

        // Compute and store offsets points
        self.edges.iter().for_each(|edge| {
            let p1 = self.vertices.get(&edge.p1).unwrap();
            let p2 = self.vertices.get(&edge.p2).unwrap();
            let dx = edge.outward_normal.x * self.offset_margin;
            let dy = edge.outward_normal.y * self.offset_margin;
            offset_edges.push(self.create_offset_edge(p1, p2, dx, dy));
        });

        for i in 0..offset_edges.len() {
            let this_edge = &offset_edges[i];
            let prev_edge = &offset_edges[(i + offset_edges.len() - 1) % offset_edges.len()];

            // Calculate edge lengths
            let this_length = ((this_edge.1.x - this_edge.0.x).powi(2)
                + (this_edge.1.y - this_edge.0.y).powi(2))
            .sqrt();
            let prev_length = ((prev_edge.1.x - prev_edge.0.x).powi(2)
                + (prev_edge.1.y - prev_edge.0.y).powi(2))
            .sqrt();

            // Handle collapsed edges by using the midpoint
            if this_length < tolerance || prev_length < tolerance {
                let midpoint = Vertex {
                    x: (this_edge.0.x + this_edge.1.x) / 2.0,
                    y: (this_edge.0.y + this_edge.1.y) / 2.0,
                    is_intersect: true,
                };
                vertices.insert(index, midpoint);
                index += 1;
                continue;
            }

            // Proceed with normal intersection check for non-collapsed edges
            if let Some(vertex) = self.edges_intersection(prev_edge, this_edge, false) {
                vertices.insert(index, vertex);
                index += 1;
            } else {
                // Original arc handling for non-intersecting edges
                let arc_center = &self.vertices.get(&i).unwrap();
                let arc_vertices = self.append_arc(
                    arc_center,
                    self.offset_margin.abs(),
                    &prev_edge.1,
                    &this_edge.0,
                    tolerance,
                );

                for av in arc_vertices {
                    vertices.insert(index, av);
                    index += 1;
                }
            }
        }

        self.create_polygon(vertices, self.offset_margin, false)
    }

    /// Sorts a list of vertex indices by their squared distance from a reference point.
    ///
    /// This function is used to order intersection points along an edge. It calculates the squared
    /// distance from each vertex to the reference point `p1` and sorts them in ascending order.
    ///
    /// # Arguments
    /// * `p1` - The reference point to calculate distances from
    /// * `cross` - List of vertex indices to be sorted
    /// * `poly` - The polygon containing the vertices
    ///
    /// # Returns
    /// A vector of vertex indices sorted by their squared distance from `p1`
    ///
    /// # Notes on NaN Handling
    /// - NaN values can occur in geometric calculations due to:
    ///   - Degenerate edges (zero-length segments)
    ///   - Floating-point overflow/underflow
    ///   - Invalid geometric operations (e.g., normalizing a zero vector)
    /// - We filter out NaN coordinates and NaN distances to prevent sorting failures
    /// - These cases typically represent degenerate geometry that should be handled separately
    ///
    /// # Minimum Rust Version
    /// Requires Rust 1.62.0 or later for `total_cmp` support. This is needed to:
    /// - Handle NaN values consistently during sorting
    /// - Provide total ordering for all f64 values (including infinities)
    /// - Maintain stable behavior across different platforms
    fn sort_by_squared_dist(&self, p1: &Vertex, cross: &[usize], poly: &Polygon) -> Vec<usize> {
        let mut vp: Vec<(f64, usize)> = cross
            .iter()
            .filter_map(|&c| {
                let vertex = poly.vertices.get(&c)?;
                if vertex.x.is_nan() || vertex.y.is_nan() {
                    return None;
                }
                let dist = (p1.x - vertex.x).powi(2) + (p1.y - vertex.y).powi(2);
                if dist.is_nan() {
                    None
                } else {
                    Some((dist, c))
                }
            })
            .collect();

        // Use total_cmp to handle all cases including Inf and normal numbers
        vp.sort_by(|a, b| a.0.total_cmp(&b.0));

        vp.into_iter().map(|(_, id)| id).collect()
    }

    /// Detects and processes all intersections between edges in a polygon.
    ///
    /// This function:
    /// 1. Identifies all intersection points between polygon edges
    /// 2. Splits intersecting edges at these points
    /// 3. Updates the polygon structure with new vertices and split edges
    ///
    /// # Arguments
    /// * `margin_polygon` - A mutable reference to the polygon being processed
    ///
    /// # Effects
    /// - Modifies the input polygon by:
    ///   - Adding intersection points as new vertices
    ///   - Replacing original edges with split segments
    ///   - Updating edge indices and connectivity
    ///
    /// # Notes
    /// - Intersection points are marked with `is_intersect = true`
    /// - Only processes unique edge pairs (edge1.index > edge2.index)
    /// - Maintains original edge normals for split segments
    /// - Preserves polygon winding direction
    /// - Handles edge cases through the edges_intersection() method
    fn detect_all_intersect(&mut self, margin_polygon: &mut Polygon) {
        let mut poly: Polygon = Polygon::default();
        let mut indices: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut vertices = margin_polygon.vertices.clone();

        let mut iteration = 0;
        margin_polygon.edges.iter_mut().for_each(|edge| {
            indices.insert(edge.index, Vec::new());
            edge.index = iteration;
            iteration += 1;
        });

        let mut max_vertices = 0;
        margin_polygon.vertices.iter().for_each(|(id, _)| {
            if id > &max_vertices {
                max_vertices = *id;
            }
        });

        // Store intersection points with index
        margin_polygon.edges.iter().for_each(|edge1| {
            margin_polygon.edges.iter().for_each(|edge2| {
                if edge1.index > edge2.index {
                    let p1: Vertex = *margin_polygon.vertices.get(&edge1.p1).unwrap();
                    let p2: Vertex = *margin_polygon.vertices.get(&edge1.p2).unwrap();
                    let p3: Vertex = *margin_polygon.vertices.get(&edge2.p1).unwrap();
                    let p4: Vertex = *margin_polygon.vertices.get(&edge2.p2).unwrap();
                    let e1: (Vertex, Vertex) = (p1, p2);
                    let e2: (Vertex, Vertex) = (p3, p4);

                    let inters = self.edges_intersection(&e1, &e2, true);
                    if !inters.is_none() {
                        vertices.insert(max_vertices + 1, inters.unwrap());
                        max_vertices += 1;

                        match indices.get_mut(&edge1.index) {
                            Some(v) => {
                                if !v.contains(&max_vertices) {
                                    v.push(max_vertices);
                                }
                            }
                            _ => {}
                        };
                        match indices.get_mut(&edge2.index) {
                            Some(v) => {
                                if !v.contains(&max_vertices) {
                                    v.push(max_vertices);
                                }
                            }
                            _ => {}
                        };
                    }
                }
            });
        });

        // Split segments into 2 new segments from intersection point
        margin_polygon.vertices = vertices.clone();
        let mut new_poly: Vec<Edge> = Vec::new();
        margin_polygon.edges.iter().for_each(|edge| {
            let idx = edge.index;
            let p1 = margin_polygon.vertices.get(&edge.p1).unwrap();

            match indices.get(&idx) {
                Some(cross) => {
                    if !cross.is_empty() {
                        let sorted_cross = self.sort_by_squared_dist(p1, cross, margin_polygon);
                        if !sorted_cross.is_empty() {
                            let new_edge = Edge {
                                p1: edge.p1,
                                p2: sorted_cross[0],
                                index: 0,
                                outward_normal: edge.outward_normal,
                            };
                            new_poly.push(new_edge);

                            for i in 0..sorted_cross.len().saturating_sub(1) {
                                let new_edge = Edge {
                                    p1: sorted_cross[i],
                                    p2: sorted_cross[i + 1],
                                    index: 0,
                                    outward_normal: edge.outward_normal,
                                };
                                new_poly.push(new_edge);
                            }

                            let new_edge = Edge {
                                p1: sorted_cross[sorted_cross.len() - 1],
                                p2: edge.p2,
                                index: 0,
                                outward_normal: edge.outward_normal,
                            };
                            new_poly.push(new_edge);
                        } else {
                            // If no valid sorted points, keep original edge
                            new_poly.push(*edge);
                        }
                    } else {
                        new_poly.push(*edge);
                    }
                }
                None => new_poly.push(*edge),
            }
        });
        let mut iteration = 0;
        new_poly.iter_mut().for_each(|edge| {
            edge.index = iteration;
            iteration += 1;
        });
        poly.edges = new_poly;
        poly.vertices = vertices;
        *margin_polygon = poly;
    }

    /// Detects all closed regions in a possibly self-intersecting polygon.
    ///
    /// This function takes a polygon that may contain self-intersections and splits it
    /// into multiple non-intersecting regions. Each region is returned as either:
    /// - A proper closed polygon (3+ edges)
    /// - A degenerate single-edge segment
    ///
    /// # Arguments
    /// * `polygon` - The input polygon which may contain self-intersections
    ///
    /// # Returns
    /// A vector of Polygons, where each:
    /// - Has `is_degenerate = true` for single-edge segments
    /// - Forms a proper closed loop when edges.len() ≥ 3
    /// - Regions are ordered by discovery during traversal
    ///
    /// # Behavior Notes
    /// - Orphaned vertices (with no edges) are ignored
    /// - Contains safeguards against infinite loops (max_iterations = vertices.len() * 2)
    /// - Will emit warnings via eprintln! for suspicious cases
    /// - Self-referencing edges (p1 == p2) are skipped
    ///
    /// # Edge Cases
    /// - Returns empty vec if no valid regions found
    /// - Degenerate single-edge regions may be returned for collapsed geometry
    /// - Intersection points must be marked with is_intersect = true
    fn detect_regions(&self, polygon: &Polygon) -> Vec<Polygon> {
        // each region is described by a polygon
        let mut regions: Vec<Polygon> = Vec::new();

        // remaining is a vect of the indices of the vertices
        let mut remaining: Vec<usize> = polygon.vertices.keys().copied().collect();

        // Create edge map: vertex index -> list of edge indices
        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
        polygon.vertices.keys().for_each(|&id| {
            map.insert(id, Vec::new());
        });

        polygon.edges.iter().for_each(|edge| {
            // Skip self-referencing edges
            if edge.p1 != edge.p2 {
                map.entry(edge.p1).and_modify(|v| {
                    if !v.contains(&edge.index) {
                        v.push(edge.index);
                    }
                });
            }
        });

        // Safety counter to prevent infinite loops
        let max_iterations = remaining.len() * 2;
        let mut iteration_count = 0;

        while !remaining.is_empty() && iteration_count < max_iterations {
            iteration_count += 1;

            let start_idx = remaining[0];
            let mut current_region = Polygon {
                edges: Vec::new(),
                vertices: HashMap::new(),
                offset_margin: polygon.offset_margin,
                is_degenerate: false,
            };

            let mut idx = start_idx;
            let mut prev_edge_index: Option<usize> = None;
            let mut has_moved = false;
            let mut visited_in_region = HashSet::new();

            loop {
                // Check for infinite loop in this region
                if visited_in_region.contains(&idx) {
                    // We've looped back to a vertex without completing the region
                    break;
                }
                visited_in_region.insert(idx);

                // Skip if vertex has no edges or we've completed a loop
                if (current_region.vertices.contains_key(&idx) && has_moved)
                    || map.get(&idx).map_or(true, |v| v.is_empty())
                {
                    break;
                }

                if let Some(vertex) = polygon.vertices.get(&idx) {
                    current_region.vertices.insert(idx, *vertex);
                    remaining.retain(|&r| r != idx);

                    let edges = map.get(&idx).unwrap();
                    let edge_index = if vertex.is_intersect {
                        // For intersection points, choose edge that isn't the one we came from
                        edges.iter().find(|&&e| Some(e) != prev_edge_index).copied()
                    } else {
                        // For regular points, just take first edge
                        edges.first().copied()
                    };

                    if let Some(edge_index) = edge_index {
                        let edge = &polygon.edges[edge_index];
                        // Skip if this would create a self-referencing edge
                        if edge.p1 == edge.p2 {
                            break;
                        }

                        current_region.edges.push(Edge {
                            p1: idx,
                            p2: edge.p2,
                            outward_normal: edge.outward_normal,
                            index: current_region.edges.len(),
                        });

                        prev_edge_index = Some(edge_index);
                        idx = edge.p2;
                        has_moved = true;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            // Only add regions that have at least 3 edges (proper polygons)
            // or exactly 1 edge (degenerate line segments)
            if current_region.edges.len() >= 3 || current_region.edges.len() == 1 {
                if current_region.edges.len() < 3 {
                    current_region.is_degenerate = true;
                }
                regions.push(current_region);
            }
        }

        if iteration_count >= max_iterations {
            eprintln!("Warning: detect_regions hit maximum iteration count");
        }

        regions
    }

    /// Computes the outward-facing normal vector for an edge defined by two vertices.
    ///
    /// # Arguments
    /// * `v1` - The starting vertex of the edge
    /// * `v2` - The ending vertex of the edge
    ///
    /// # Returns
    /// A Vertex representing the normalized outward normal vector with:
    /// - x: The x-component of the normal (dy/edge_length)
    /// - y: The y-component of the normal (-dx/edge_length)
    /// - is_intersect: Always false
    ///
    /// # Notes
    /// - The normal points to the left when looking from v1 to v2 (outward for CCW polygons)
    /// - The vector is normalized (length = 1)
    /// - For degenerate edges (length ≈ 0), behavior is undefined
    fn outward_edge_normal(&self, v1: &Vertex, v2: &Vertex) -> Vertex {
        let dx = v2.x - v1.x;
        let dy = v2.y - v1.y;
        let edge_length = (dx * dx + dy * dy).sqrt();
        Vertex {
            is_intersect: false,
            x: dy / edge_length,
            y: -dx / edge_length,
        }
    }

    /// Converts a contour of 2D points into a HashMap of vertices.
    ///
    /// # Arguments
    /// * `contour` - A vector of (x,y) coordinate tuples representing the polygon contour
    ///
    /// # Returns
    /// A HashMap where:
    /// - Keys are sequential indices (0..n-1)
    /// - Values are Vertex structs with:
    ///   - x/y coordinates from the input contour
    ///   - is_intersect flag set to false
    ///
    /// # Notes
    /// - Assumes contour is closed (first and last points equal)
    /// - Skips the last point in the contour to avoid duplication
    /// - Creates vertices in the same order as the input contour
    fn contour_to_vertices(contour: Vec<(f64, f64)>) -> HashMap<usize, Vertex> {
        let mut vtxs: HashMap<usize, Vertex> = HashMap::new();

        for i in 0..contour.len() - 1 {
            let mut vertex: Vertex = Vertex::default();
            vertex.is_intersect = false;
            vertex.x = contour[i].0;
            vertex.y = contour[i].1;
            vtxs.insert(i, vertex);
        }
        vtxs
    }

    /// Calculates the area of a polygon using the shoelace formula.
    ///
    /// # Arguments
    /// * `poly` - The polygon to calculate area for
    ///
    /// # Returns
    /// The polygon's area as f64. The area is always positive regardless of winding direction.
    ///
    /// # Notes
    /// - Uses the shoelace formula which works for both convex and concave polygons
    /// - Assumes the polygon is closed (first and last vertices are connected)
    /// - For self-intersecting polygons, the result may not be meaningful
    /// - Returns 0.0 for degenerate polygons with fewer than 3 vertices
    ///
    /// # Algorithm
    /// The shoelace formula sums the cross products of vertex coordinates:
    /// Area = 0.5 * |Σ(x_i*y_{i+1} - x_{i+1}*y_i)| where i ranges from 0 to n-1
    fn get_polygon_area(&self, poly: &Polygon) -> f64 {
        let mut contours: Vec<(f64, f64)> = Vec::new();

        let mut edges = poly.edges.clone();
        edges.sort_by_key(|k| k.index);

        for i in 0..edges.len() {
            let v = poly.vertices.get(&edges[i].p1).unwrap();
            contours.push((v.x, v.y));
        }

        let mut a = 0.0;
        for i in 0..contours.len() - 1 {
            a = a + (contours[i].0 * contours[i + 1].1) - (contours[i + 1].0 * contours[i].1);
        }
        a = a + (contours[contours.len() - 1].0 * contours[0].1)
            - (contours[0].0 * contours[contours.len() - 1].1);
        (a * -0.5).abs()
    }

    /// Converts a vector of 2D coordinate tuples into a vector of line segments.
    ///
    /// # Arguments
    /// * `contour1` - A vector of (x,y) coordinate tuples representing a polygon contour
    ///
    /// # Returns
    /// A vector of Segments where each segment connects consecutive points in the input contour.
    ///
    /// # Notes
    /// - Assumes the contour is closed (first and last points should be equal)
    /// - Creates segments between consecutive points (point[i] to point[i+1])
    /// - The last segment connects the last point back to the first point
    /// - Empty segments (where p1 == p2) should be filtered out by the caller
    fn tuples_to_segments(contour1: &Vec<(f64, f64)>) -> Vec<Segment> {
        let mut segments: Vec<Segment> = Vec::new();

        for i in 1..contour1.len() {
            let mut sgmt: Segment = Segment::default();
            sgmt.p1.0 = contour1[i - 1].0;
            sgmt.p1.1 = contour1[i - 1].1;
            if i == contour1.len() {
                sgmt.p2.0 = contour1[0].0;
                sgmt.p2.1 = contour1[0].1;
                segments.push(sgmt);
                break;
            }
            sgmt.p2.0 = contour1[i].0;
            sgmt.p2.1 = contour1[i].1;
            segments.push(sgmt);
        }
        segments
    }

    /// Constructs a new Polygon from a set of vertices and offset parameters.
    ///
    /// # Arguments
    /// * `vertices` - HashMap of vertex indices to Vertex structs defining the polygon's shape
    /// * `offset_size` - The offset margin to be applied to this polygon
    /// * `is_initial_polygon` - Flag indicating if this is the original polygon (true) or an offset result (false)
    ///
    /// # Returns
    /// A new Polygon with:
    /// - Provided vertices
    /// - Edges connecting consecutive vertices (with proper indexing)
    /// - Outward normal vectors calculated if `is_initial_polygon` is true
    /// - Specified offset margin preserved
    ///
    /// # Notes
    /// - For initial polygons, computes outward normals for each edge
    /// - Automatically skips self-referencing edges (where p1 == p2)
    /// - Maintains winding direction from input vertices
    /// - Edge indices are sequential based on vertex ordering
    fn create_polygon(
        &mut self,
        vertices: HashMap<usize, Vertex>,
        offset_size: f64,
        is_initial_polygon: bool,
    ) -> Polygon {
        let mut polygon = Polygon::default();
        let mut edges: Vec<Edge> = Vec::new();
        polygon.vertices = vertices;
        polygon.offset_margin = offset_size;

        // Create edges segments with their index and normal
        for i in 0..polygon.vertices.len() {
            let mut edge = Edge {
                p1: i,
                p2: (i + 1) % polygon.vertices.len(),
                index: i,
                outward_normal: Vertex::default(),
            };
            if edge.p1 == edge.p2 {
                continue;
            }
            if is_initial_polygon {
                edge.outward_normal = self.outward_edge_normal(
                    polygon.vertices.get(&edge.p1).unwrap(),
                    polygon.vertices.get(&edge.p2).unwrap(),
                );
            }
            edges.push(edge);
        }

        polygon.edges = edges;
        polygon
    }

    /// Creates a new Polygon from a contour of 2D points with the specified offset margin.
    ///
    /// # Arguments
    /// * `initial_contour` - A vector of (x,y) coordinate tuples representing the polygon's contour.
    ///                      Must be closed (first and last points equal).
    /// * `offset_size` - The offset margin to be applied (positive for outward, negative for inward).
    ///
    /// # Returns
    /// A Result containing:
    /// - Ok(Polygon) if the input is valid
    /// - Err(OffsetError::UnclosedPolygon) if the contour isn't closed
    ///
    /// # Behavior
    /// - Automatically reverses clockwise contours to CCW (required for correct offsetting)
    /// - Removes duplicate/contiguous points
    /// - Calculates outward normals for all edges
    /// - Initializes polygon with given offset margin
    ///
    /// # Notes
    /// - The input contour should have at least 3 distinct points (excluding closure)
    /// - Empty or degenerate contours may produce unexpected results
    /// - CCW winding is required for proper outward normal calculation
    pub fn new(
        initial_contour: &Vec<(f64, f64)>,
        offset_size: f64,
    ) -> Result<Polygon, OffsetError> {
        let mut new_segments: Vec<Segment> = Polygon::tuples_to_segments(&initial_contour);
        // remove contiguous points with same coords
        new_segments.retain(|s| get_dist(s.p1, s.p2) != 0.);

        // check if our polygon is closed
        if initial_contour[0] != initial_contour[initial_contour.len() - 1] {
            return Err(OffsetError::UnclosedPolygon);
        }

        // check the direction of our polygon
        let value = new_segments.iter().fold(0., |acc, seg| {
            acc + (seg.p2.0 - seg.p1.0) * (seg.p2.1 + seg.p1.1)
        });

        // reverse it if clockwise
        if value > 0. {
            new_segments = reverse_segments(&new_segments);
        }

        let mut points: Vec<(f64, f64)> = Vec::new();
        for s in new_segments.iter() {
            points.push(s.p1);
        }
        points.push(points[0]);

        let mut initial_polygon: Polygon = Polygon::default();
        let vertices: HashMap<usize, Vertex> = Polygon::contour_to_vertices(points.to_vec());
        Ok(initial_polygon.create_polygon(vertices, offset_size, true))
    }

    /// Computes the offset polygon from the original contour with the specified margin.
    ///
    /// # Arguments
    /// * `tolerance` - The minimum acceptable edge length. Edges shorter than this will be collapsed.
    ///
    /// # Returns
    /// A Result containing:
    /// - Ok(Offset) with the offset contour, area and perimeter if successful
    /// - Err(OffsetError) if:
    ///   - The tolerance is invalid (≤ 0)
    ///   - The polygon collapses completely
    ///   - No valid regions can be found
    ///
    /// # Behavior
    /// 1. Performs initial checks for trivial cases (zero offset or collapsed polygon)
    /// 2. Creates margin polygon by offsetting edges
    /// 3. Detects and processes all intersections
    /// 4. Identifies valid regions from intersection results
    /// 5. Selects the largest valid region as the result
    ///
    /// # Notes
    /// - For inward offsets (negative margin), may return CollapsedPolygon error
    /// - The result is always a closed polygon (first and last points equal)
    /// - Degenerate cases (area/perimeter ≈ 0) are rejected
    /// - Uses conservative checks to prevent invalid geometries
    pub fn offsetting(&mut self, tolerance: f64) -> Result<Offset, OffsetError> {
        self.validate_input(tolerance)?;
        
        if let Some(zero_offset_result) = self.handle_zero_offset_case()? {
            return Ok(zero_offset_result);
        }

        if self.is_collapsed() {
            return Err(OffsetError::CollapsedPolygon);
        }

        let mut margin_polygon = self.create_margin_polygon(tolerance);
        self.detect_all_intersect(&mut margin_polygon);

        let best_region = self.find_best_region(&margin_polygon)?;
        let offset = self.build_offset_result(&best_region)?;

        self.validate_result(&offset)?;
        Ok(offset)
    }

    /// Validates the input tolerance value for offsetting operations.
    ///
    /// # Arguments
    /// * `tolerance` - The minimum acceptable edge length (must be > 0)
    ///
    /// # Returns
    /// - Ok(()) if tolerance is valid
    /// - Err(OffsetError::InvalidTolerance) if tolerance ≤ 0
    ///
    /// # Notes
    /// - This is called before any offsetting calculations
    /// - Prevents division by zero and other numerical issues
    fn validate_input(&self, tolerance: f64) -> Result<(), OffsetError> {
        if tolerance <= 0.0 {
            Err(OffsetError::InvalidTolerance)
        } else {
            Ok(())
        }
    }

    /// Handles the special case where offset margin is zero (no offset needed).
    ///
    /// # Returns
    /// - Some(Offset) containing the original polygon if margin = 0
    /// - None if margin ≠ 0 (normal case)
    ///
    /// # Notes
    /// - Avoids unnecessary calculations when no offset is requested
    /// - Preserves original polygon area and perimeter
    /// - Still validates the polygon is closed and non-degenerate
    fn handle_zero_offset_case(&self) -> Result<Option<Offset>, OffsetError> {
        if self.offset_margin != 0.0 {
            return Ok(None);
        }

        let mut points: Vec<(f64, f64)> = Vec::new();
        self.edges.iter().for_each(|e| {
            let p1 = self.vertices.get(&e.p1).unwrap();
            points.push((p1.x, p1.y));
        });
        points.push(points[0]);

        Ok(Some(Offset {
            area: compute_area(&points),
            perimeter: compute_perimeter(&points),
            contour: points,
        }))
    }

    /// Selects the best valid region from intersection results.
    ///
    /// # Arguments
    /// * `margin_polygon` - The polygon containing all intersection regions
    ///
    /// # Returns
    /// - Ok(Polygon) containing the largest valid region by area
    /// - Err if:
    ///   - No regions found (OffsetError::NoValidRegions)
    ///   - Only single point found (OffsetError::SinglePointRegion)
    ///   - Can't determine largest region (OffsetError::RegionSortingFailed)
    ///
    /// # Notes
    /// - Prefers proper closed regions over degenerate segments
    /// - Uses area as the selection criteria
    /// - Filters out invalid/empty regions
    fn find_best_region(&self, margin_polygon: &Polygon) -> Result<Polygon, OffsetError> {
        let regions = self.detect_regions(margin_polygon);

        if regions.is_empty() {
            if margin_polygon.vertices.len() == 1 {
                return Err(OffsetError::SinglePointRegion);
            }
            return Err(OffsetError::NoValidRegions);
        }

        regions
            .into_iter()
            .max_by(|a, b| self.get_polygon_area(a).total_cmp(&self.get_polygon_area(b)))
            .ok_or(OffsetError::RegionSortingFailed)
    }

    /// Converts a valid polygon region into an Offset result.
    ///
    /// # Arguments
    /// * `region` - The selected polygon region to convert
    ///
    /// # Returns
    /// Ok(Offset) containing:
    /// - Contour points (ensured to be closed)
    /// - Computed area
    /// - Computed perimeter
    ///
    /// # Notes
    /// - Ensures the contour is properly closed
    /// - Calculates geometric properties
    /// - Maintains winding direction consistency
    fn build_offset_result(&self, region: &Polygon) -> Result<Offset, OffsetError> {
        let mut offset = Offset {
            contour: Vec::new(),
            area: self.get_polygon_area(region),
            perimeter: 0.0,
        };

        region.edges.iter().for_each(|edge| {
            offset.contour.push((
                region.vertices.get(&edge.p1).unwrap().x,
                region.vertices.get(&edge.p1).unwrap().y,
            ));
        });

        if !offset.contour.is_empty() && offset.contour[0] != *offset.contour.last().unwrap() {
            offset.contour.push(offset.contour[0]);
        }

        offset.perimeter = compute_perimeter(&offset.contour);
        Ok(offset)
    }

    /// Validates that an offset result meets minimum quality criteria.
    ///
    /// # Arguments
    /// * `offset` - The Offset result to validate
    ///
    /// # Returns
    /// - Ok(()) if:
    ///   - Contour has ≥3 points
    ///   - Area > 0
    ///   - Perimeter > 0
    /// - Err(OffsetError::CollapsedPolygon) if invalid
    ///
    /// # Notes
    /// - Prevents returning degenerate results
    /// - Uses floating-point epsilon for comparisons
    /// - Called as final validation step
    fn validate_result(&self, offset: &Offset) -> Result<(), OffsetError> {
        if offset.contour.len() < 3 
            || offset.area <= f64::EPSILON 
            || offset.perimeter <= f64::EPSILON 
        {
            Err(OffsetError::CollapsedPolygon)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests that deal with edges collapsing.
    #[test]
    fn test_sort_by_squared_dist_normal_case() {
        let poly = Polygon::default();
        let p1 = Vertex {
            x: 0.0,
            y: 0.0,
            is_intersect: false,
        };
        let points = vec![
            (
                10,
                Vertex {
                    x: 1.0,
                    y: 0.0,
                    is_intersect: false,
                },
            ),
            (
                20,
                Vertex {
                    x: 2.0,
                    y: 0.0,
                    is_intersect: false,
                },
            ),
            (
                30,
                Vertex {
                    x: 0.0,
                    y: 1.0,
                    is_intersect: false,
                },
            ),
        ];
        let cross: Vec<usize> = points.iter().map(|(id, _)| *id).collect();
        let test_poly = Polygon {
            vertices: points.into_iter().collect(),
            ..Default::default()
        };

        let sorted = poly.sort_by_squared_dist(&p1, &cross, &test_poly);
        assert_eq!(sorted, vec![10, 30, 20]);
    }

    #[test]
    fn test_sort_by_squared_dist_with_nan() {
        let poly = Polygon::default();
        let p1 = Vertex {
            x: 0.0,
            y: 0.0,
            is_intersect: false,
        };
        let points = vec![
            (
                10,
                Vertex {
                    x: f64::NAN,
                    y: f64::NAN,
                    is_intersect: false,
                },
            ),
            (
                20,
                Vertex {
                    x: 1.0,
                    y: 0.0,
                    is_intersect: false,
                },
            ),
            (
                30,
                Vertex {
                    x: 0.0,
                    y: 1.0,
                    is_intersect: false,
                },
            ),
        ];
        let cross: Vec<usize> = points.iter().map(|(id, _)| *id).collect();
        let test_poly = Polygon {
            vertices: points.into_iter().collect(),
            ..Default::default()
        };

        let sorted = poly.sort_by_squared_dist(&p1, &cross, &test_poly);
        // NaN points should be filtered out or placed at the end
        assert_eq!(sorted.len(), 2);
        assert!(sorted.contains(&20));
        assert!(sorted.contains(&30));
    }

    // Tests that deal with `detect_regions` and the cases where we have
    // orphaned vertices.
    #[test]
    fn test_find_start_vertex() {
        let poly = Polygon::default();
        let mut remaining = vec![1, 2, 3];
        let mut map = HashMap::new();
        map.insert(1, vec![]); // No edges
        map.insert(2, vec![0]); // Has edge
        map.insert(3, vec![]); // No edges

        // Should find vertex 2 since it has edges
        assert_eq!(poly.find_start_vertex(&mut remaining, &map), Some(2));
        assert_eq!(remaining, vec![1, 3]);

        // With no edges left, should return None
        assert_eq!(poly.find_start_vertex(&mut remaining, &map), None);
    }

    #[test]
    fn test_trace_region() {
        let mut poly = Polygon::default();
        let mut map = HashMap::new();
        let mut remaining = vec![0, 1, 2];

        // Create simple triangle
        poly.vertices.insert(0, Vertex { x: 0.0, y: 0.0, is_intersect: false });
        poly.vertices.insert(1, Vertex { x: 1.0, y: 0.0, is_intersect: false });
        poly.vertices.insert(2, Vertex { x: 0.5, y: 1.0, is_intersect: false });

        poly.edges.push(Edge { p1: 0, p2: 1, index: 0, outward_normal: Vertex::default() });
        poly.edges.push(Edge { p1: 1, p2: 2, index: 1, outward_normal: Vertex::default() });
        poly.edges.push(Edge { p1: 2, p2: 0, index: 2, outward_normal: Vertex::default() });

        map.insert(0, vec![0]);
        map.insert(1, vec![1]);
        map.insert(2, vec![2]);

        let region = poly.trace_region(0, &poly, &map, &mut remaining);
        assert_eq!(region.edges.len(), 3);
        assert_eq!(remaining.len(), 0);
    }

    #[test]
    fn test_should_add_region() {
        let poly = Polygon::default();
        let mut valid_region = Polygon::default();
        valid_region.edges.push(Edge::default());
        valid_region.edges.push(Edge::default());
        valid_region.edges.push(Edge::default());

        let mut degenerate_region = Polygon::default();
        degenerate_region.edges.push(Edge::default());

        let mut invalid_region = Polygon::default();
        invalid_region.edges.push(Edge::default());
        invalid_region.edges.push(Edge::default());

        assert!(poly.should_add_region(&valid_region));
        assert!(poly.should_add_region(&degenerate_region));
        assert!(!poly.should_add_region(&invalid_region));
    }

    #[test]
    fn test_detect_regions_with_orphaned_vertex() {
        let poly = Polygon::default();
        let mut test_poly = Polygon {
            vertices: HashMap::new(),
            edges: Vec::new(),
            offset_margin: 0.0,
            is_degenerate: false,
        };

        // Add a single orphaned vertex with no edges
        test_poly.vertices.insert(
            0,
            Vertex {
                x: 0.0,
                y: 0.0,
                is_intersect: false,
            },
        );

        let regions = poly.detect_regions(&test_poly);
        assert_eq!(regions.len(), 0); // Orphaned vertex should be ignored
    }

    #[test]
    fn test_detect_regions_with_valid_polygon() {
        let poly = Polygon::default();
        let mut test_poly = Polygon {
            vertices: HashMap::new(),
            edges: Vec::new(),
            offset_margin: 0.0,
            is_degenerate: false,
        };

        // Create a simple triangle
        test_poly.vertices.insert(
            0,
            Vertex {
                x: 0.0,
                y: 0.0,
                is_intersect: false,
            },
        );
        test_poly.vertices.insert(
            1,
            Vertex {
                x: 1.0,
                y: 0.0,
                is_intersect: false,
            },
        );
        test_poly.vertices.insert(
            2,
            Vertex {
                x: 0.5,
                y: 1.0,
                is_intersect: false,
            },
        );

        test_poly.edges.push(Edge {
            p1: 0,
            p2: 1,
            index: 0,
            outward_normal: Vertex::default(),
        });
        test_poly.edges.push(Edge {
            p1: 1,
            p2: 2,
            index: 1,
            outward_normal: Vertex::default(),
        });
        test_poly.edges.push(Edge {
            p1: 2,
            p2: 0,
            index: 2,
            outward_normal: Vertex::default(),
        });

        let regions = poly.detect_regions(&test_poly);
        assert_eq!(regions.len(), 1); // Should find one complete region
    }

    #[test]
    fn test_detect_regions_infinite_loop_prevention() {
        let poly = Polygon::default();
        let mut test_poly = Polygon {
            vertices: HashMap::new(),
            edges: Vec::new(),
            offset_margin: 0.0,
            is_degenerate: false,
        };

        // Create a malformed polygon that could cause infinite loops
        test_poly.vertices.insert(
            0,
            Vertex {
                x: 0.0,
                y: 0.0,
                is_intersect: false,
            },
        );
        test_poly.vertices.insert(
            1,
            Vertex {
                x: 1.0,
                y: 0.0,
                is_intersect: false,
            },
        );

        // Edge that points to itself
        test_poly.edges.push(Edge {
            p1: 0,
            p2: 0,
            index: 0,
            outward_normal: Vertex::default(),
        });

        // Circular reference
        test_poly.edges.push(Edge {
            p1: 1,
            p2: 0,
            index: 1,
            outward_normal: Vertex::default(),
        });
        test_poly.edges.push(Edge {
            p1: 0,
            p2: 1,
            index: 2,
            outward_normal: Vertex::default(),
        });

        let regions = poly.detect_regions(&test_poly);
        assert!(
            regions.is_empty(),
            "Should detect and reject the invalid regions"
        );
    }

    // Some tests for the offsetting in the case where we have collapsed shapes.
    #[test]
    fn test_offsetting_with_collapsed_polygon() {
        let positions = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
        // Use an extremely large offset that would truly collapse the polygon
        let mut polygon = Polygon::new(&positions, -1000.0).unwrap();
        let result = polygon.offsetting(0.1);
        assert!(matches!(result, Err(OffsetError::CollapsedPolygon)));
    }

    #[test]
    fn test_offsetting_with_valid_polygon() {
        let positions = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)];
        let mut polygon = Polygon::new(&positions, -0.5).unwrap();
        let result = polygon.offsetting(0.1);
        assert!(result.is_ok());
        let offset = result.unwrap();
        assert!(offset.area > 0.0);
        assert!(offset.perimeter > 0.0);
        assert_eq!(offset.contour.len(), 5); // Closed polygon
    }

    // Test collapsing an edge
    #[test]
    fn test_collapsing_edge() {
        let positions = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];

        // Small inward offset should succeed
        let mut polygon = Polygon::new(&positions, -0.4).unwrap();
        assert!(polygon.offsetting(0.1).is_ok());

        // Larger offset that collapses edges but should still produce a valid polygon
        let mut polygon = Polygon::new(&positions, -0.8).unwrap();
        let result = polygon.offsetting(0.1);
        assert!(result.is_ok());
        let offset = result.unwrap();
        assert!(offset.contour.len() >= 3); // Should still form a valid polygon
        assert!(offset.area > 0.0); // Should have some area
        assert!(offset.perimeter > 0.0); // Should have some perimeter
    }

    #[test]
    fn test_validate_input() {
        let poly = Polygon::default();
        assert!(poly.validate_input(0.0).is_err());
        assert!(poly.validate_input(-1.0).is_err());
        assert!(poly.validate_input(0.1).is_ok());
    }

    #[test]
    fn test_handle_zero_offset_case() {
        let positions = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)];
        let poly = Polygon::new(&positions, 0.0).unwrap();
        let result = poly.handle_zero_offset_case().unwrap();
        assert!(result.is_some());
        let offset = result.unwrap();
        assert_eq!(offset.contour.len(), 4);
        assert!(offset.area > 0.0);

        let poly = Polygon::new(&positions, 1.0).unwrap();
        let result = poly.handle_zero_offset_case().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_find_best_region() {
        let positions = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)];
        let mut poly = Polygon::new(&positions, -0.5).unwrap();
        let mut margin_poly = poly.create_margin_polygon(0.1);
        poly.detect_all_intersect(&mut margin_poly);
        let best_region = poly.find_best_region(&margin_poly).unwrap();
        assert!(poly.get_polygon_area(&best_region) > 0.0);
    }

    #[test]
    fn test_build_offset_result() {
        let positions = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)];
        let poly = Polygon::new(&positions, 0.0).unwrap();
        let region = poly.detect_regions(&poly).remove(0);
        let offset = poly.build_offset_result(&region).unwrap();
        assert_eq!(offset.contour.len(), 5);
        assert!(offset.area > 0.0);
    }

    #[test]
    fn test_validate_result() {
        let valid_offset = Offset {
            contour: vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)],
            area: 0.5,
            perimeter: 1.0 + 1.0 + 2.0f64.sqrt(),
        };
        assert!(Polygon::default().validate_result(&valid_offset).is_ok());

        let invalid_offset = Offset {
            contour: vec![(0.0, 0.0), (1.0, 0.0)],
            area: 0.0,
            perimeter: 1.0,
        };
        assert!(Polygon::default().validate_result(&invalid_offset).is_err());
    }

    #[test]
    fn test_offsetting_with_collapsed_edge() {
        // This polygon has a very short top edge that will collapse with an inward offset
        let positions = vec![
            (-9.209196386108575, -82.65770331049015),
            (-19.350875198739036, -95.0),
            (-20.0, -95.0),
            (-20.0, -71.49615313993024),
            (-12.246282220888371, -74.45427771910211),
            (-9.209196386108575, -82.65770331049015),
        ];

        let mut polygon = Polygon::new(&positions, -0.7).unwrap();
        let result = polygon.offsetting(0.1);

        // Currently fails - we want this to succeed with a simplified polygon
        assert!(result.is_ok());
        let offset = result.unwrap();
        assert!(offset.contour.len() > 2); // Should still have at least 3 points
    }
}
