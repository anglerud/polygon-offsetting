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

fn compute_perimeter(contour2d: &Vec<(f64, f64)>) -> f64 {
    let mut perimeter2d = 0.;
    for i in 0..(contour2d.len() - 1) {
        let p1 = &contour2d[i];
        let p2 = &contour2d[i + 1];
        perimeter2d += ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt();
    }
    perimeter2d
}

#[inline]
fn get_dist(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p2.0 - p1.0).powi(2) + (p2.1 - p1.1).powi(2)).sqrt()
}

#[inline]
fn vector_sub(v1: (f64, f64), v2: (f64, f64)) -> (f64, f64) {
    (v1.0 - v2.0, v1.1 - v2.1)
}

#[inline]
fn vector_add(v1: (f64, f64), v2: (f64, f64)) -> (f64, f64) {
    (v1.0 + v2.0, v1.1 + v2.1)
}

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

    fn create_offset_edge(&self, p1: &Vertex, p2: &Vertex, dx: f64, dy: f64) -> (Vertex, Vertex) {
        let mut v1: Vertex = Vertex::default();
        let mut v2: Vertex = Vertex::default();
        v1.x = p1.x + dx;
        v1.y = p1.y + dy;
        v2.x = p2.x + dx;
        v2.y = p2.y + dy;
        (v1, v2)
    }

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

    // convert tuples to a Vec of Segment
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

    pub fn offsetting(&mut self, tolerance: f64) -> Result<Offset, OffsetError> {
        if tolerance <= 0.0 {
            return Err(OffsetError::InvalidTolerance);
        }

        // Early collapse detection for extreme cases
        if self.is_collapsed() {
            return Err(OffsetError::CollapsedPolygon);
        }

        if self.offset_margin == 0.0 {
            let mut points: Vec<(f64, f64)> = Vec::new();
            self.edges.iter().for_each(|e| {
                let p1 = self.vertices.get(&e.p1).unwrap();
                points.push((p1.x, p1.y));
            });
            points.push(points[0]);

            return Ok(Offset {
                area: compute_area(&points),
                perimeter: compute_perimeter(&points),
                contour: points,
            });
        }

        let mut margin_polygon = self.create_margin_polygon(tolerance);
        self.detect_all_intersect(&mut margin_polygon);

        let regions = self.detect_regions(&margin_polygon);

        // Handle truly collapsed cases
        if regions.is_empty() {
            if margin_polygon.vertices.len() == 1 {
                let pt = margin_polygon.vertices.values().next().unwrap();
                return Ok(Offset {
                    contour: vec![(pt.x, pt.y), (pt.x, pt.y)],
                    area: 0.0,
                    perimeter: 0.0,
                });
            }
            return Err(OffsetError::CollapsedPolygon);
        }

        // Find best region (largest area)
        let best_region = regions
            .iter()
            .max_by(|a, b| {
                self.get_polygon_area(a)
                    .total_cmp(&self.get_polygon_area(b))
            })
            .ok_or(OffsetError::NoValidRegions)?;

        let mut offset = Offset {
            contour: Vec::new(),
            area: self.get_polygon_area(best_region),
            perimeter: 0.0,
        };

        best_region.edges.iter().for_each(|edge| {
            offset.contour.push((
                best_region.vertices.get(&edge.p1).unwrap().x,
                best_region.vertices.get(&edge.p1).unwrap().y,
            ));
        });

        if !offset.contour.is_empty() && offset.contour[0] != *offset.contour.last().unwrap() {
            offset.contour.push(offset.contour[0]);
        }

        offset.perimeter = compute_perimeter(&offset.contour);

        // Final check - allow very small but valid polygons
        if offset.contour.len() < 3
            || offset.area <= f64::EPSILON
            || offset.perimeter <= f64::EPSILON
        {
            Err(OffsetError::CollapsedPolygon)
        } else {
            Ok(offset)
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
