use crate::BuildError;
use fluxel_ibm::BoundarySurface;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::Path,
};

/// Read an STL surface. A file whose first word is `solid` is ASCII: the text
/// after each `solid` is that solid's patch name, an empty name is `_default`,
/// and repeated names share one patch ID in first-seen order. Every solid is
/// read. Binary STL has no solid names and uses a single `_default` patch.
/// Triangle order defines anchor IDs. Identical ASCII vertex coordinates are
/// welded, and coordinates are not rescaled.
/// Errors on I/O, parsing, u32 index overflow, or invalid/degenerate surface data.
pub fn load_stl(path: &Path) -> Result<BoundarySurface, BuildError> {
    let mut file = File::open(path).map_err(BuildError::Io)?;
    let ascii = is_ascii_stl(&mut file)?;
    file.seek(SeekFrom::Start(0)).map_err(BuildError::Io)?;
    if ascii {
        load_ascii_stl(&mut file)
    } else {
        load_binary_stl(&mut file)
    }
}

/// True when the first word is `solid`. Invalid UTF-8 is treated as binary.
/// The reader may consume past the first line; the caller rewinds.
fn is_ascii_stl(file: &mut File) -> Result<bool, BuildError> {
    let mut header = String::new();
    let read = {
        let mut reader = BufReader::new(&mut *file);
        reader.read_line(&mut header)
    };
    file.seek(SeekFrom::Start(0)).map_err(BuildError::Io)?;
    match read {
        Ok(_) => Ok(header.split_whitespace().next() == Some("solid")),
        Err(error) if error.kind() == std::io::ErrorKind::InvalidData => Ok(false),
        Err(error) => Err(BuildError::Io(error)),
    }
}

fn load_binary_stl(file: &mut File) -> Result<BoundarySurface, BuildError> {
    let stl = stl_io::read_stl(file).map_err(|error| BuildError::Parse(error.to_string()))?;
    let vertices: Vec<[f64; 3]> = stl
        .vertices
        .into_iter()
        .map(|vertex| [vertex[0] as f64, vertex[1] as f64, vertex[2] as f64])
        .collect();
    if vertices.len() > u32::MAX as usize {
        return Err(BuildError::InvalidInput(
            "surface exceeds u32 vertex index capacity".into(),
        ));
    }
    let indices: Vec<[u32; 3]> = stl
        .faces
        .into_iter()
        .map(|face| {
            [
                face.vertices[0] as u32,
                face.vertices[1] as u32,
                face.vertices[2] as u32,
            ]
        })
        .collect();
    let patch_names = vec!["_default".to_string()];
    let anchor_to_patch_id = vec![0; indices.len()];
    BoundarySurface::new(&vertices, &indices, patch_names, anchor_to_patch_id)
        .map_err(BuildError::from)
}

struct ParsedStl {
    vertices: Vec<[f64; 3]>,
    indices: Vec<[u32; 3]>,
    patch_names: Vec<String>,
    anchor_to_patch_id: Vec<usize>,
}

fn load_ascii_stl(file: &mut File) -> Result<BoundarySurface, BuildError> {
    let parsed = parse_ascii_stl(file)?;
    BoundarySurface::new(
        &parsed.vertices,
        &parsed.indices,
        parsed.patch_names,
        parsed.anchor_to_patch_id,
    )
    .map_err(BuildError::from)
}

/// Read every ASCII solid. Patch IDs follow first-seen names, and vertex
/// identity matches `stl_io` by comparing `f32` bit patterns.
fn parse_ascii_stl(file: &mut File) -> Result<ParsedStl, BuildError> {
    let mut cursor = AsciiCursor::new(file);
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut anchor_to_patch_id = Vec::new();
    let mut patch_name_map = HashMap::<String, usize>::new();
    let mut vertex_index = HashMap::<[u32; 3], u32>::new();
    while let Some((line, tokens)) = cursor.next_tokens()? {
        if tokens.first().map(String::as_str) != Some("solid") {
            return Err(BuildError::Parse(format!(
                "line {line}: expected solid, got {tokens:?}"
            )));
        }
        let name = solid_patch_name(&tokens);
        loop {
            let (line, tokens) = cursor.next_tokens()?.ok_or_else(|| {
                BuildError::Parse("unexpected EOF while expecting facet or endsolid".into())
            })?;
            if tokens.first().map(String::as_str) == Some("endsolid") {
                break;
            }
            let corners = parse_facet(&mut cursor, line, &tokens)?;
            let mut triangle = [0; 3];
            for (slot, corner) in triangle.iter_mut().zip(corners) {
                *slot = intern_vertex(&mut vertices, &mut vertex_index, corner)?;
            }
            indices.push(triangle);
            anchor_to_patch_id.push(patch_id(&mut patch_name_map, &name));
        }
    }
    let mut patch_entries: Vec<(String, usize)> = patch_name_map.into_iter().collect();
    patch_entries.sort_by_key(|(_, id)| *id);
    let patch_names = patch_entries.into_iter().map(|(name, _)| name).collect();
    Ok(ParsedStl {
        vertices,
        indices,
        patch_names,
        anchor_to_patch_id,
    })
}

fn solid_patch_name(tokens: &[String]) -> String {
    if tokens.len() < 2 {
        "_default".to_string()
    } else {
        tokens[1..].join(" ")
    }
}

fn patch_id(names: &mut HashMap<String, usize>, name: &str) -> usize {
    if let Some(&id) = names.get(name) {
        return id;
    }
    let id = names.len();
    names.insert(name.to_string(), id);
    id
}

fn intern_vertex(
    vertices: &mut Vec<[f64; 3]>,
    index_of: &mut HashMap<[u32; 3], u32>,
    position: [f32; 3],
) -> Result<u32, BuildError> {
    let key = position.map(f32::to_bits);
    if let Some(&index) = index_of.get(&key) {
        return Ok(index);
    }
    let index = u32::try_from(vertices.len()).map_err(|_| {
        BuildError::InvalidInput("surface exceeds u32 vertex index capacity".into())
    })?;
    vertices.push([position[0] as f64, position[1] as f64, position[2] as f64]);
    index_of.insert(key, index);
    Ok(index)
}

fn parse_facet(
    cursor: &mut AsciiCursor<impl std::io::Read>,
    line: usize,
    header: &[String],
) -> Result<[[f32; 3]; 3], BuildError> {
    if header.len() != 5 || header[0] != "facet" || header[1] != "normal" {
        return Err(BuildError::Parse(format!(
            "line {line}: invalid facet header: {header:?}"
        )));
    }
    parse_f32_3(line, &header[2..5])?;
    expect_exact(cursor, &["outer", "loop"])?;
    let mut corners = [[0.0; 3]; 3];
    for corner in &mut corners {
        let (vertex_line, tokens) = cursor.require("vertex")?;
        if tokens.len() != 4 || tokens[0] != "vertex" {
            return Err(BuildError::Parse(format!(
                "line {vertex_line}: expected vertex x y z, got {tokens:?}"
            )));
        }
        *corner = parse_f32_3(vertex_line, &tokens[1..4])?;
    }
    expect_exact(cursor, &["endloop"])?;
    expect_exact(cursor, &["endfacet"])?;
    Ok(corners)
}

fn parse_f32_3(line: usize, tokens: &[String]) -> Result<[f32; 3], BuildError> {
    let mut values = [0.0; 3];
    for (slot, token) in values.iter_mut().zip(tokens) {
        let value = token
            .parse::<f32>()
            .map_err(|error| BuildError::Parse(format!("line {line}: {error}")))?;
        if !value.is_finite() {
            return Err(BuildError::Parse(format!(
                "line {line}: expected a finite number, got {value}"
            )));
        }
        *slot = value;
    }
    Ok(values)
}

fn expect_exact(
    cursor: &mut AsciiCursor<impl std::io::Read>,
    expected: &[&str],
) -> Result<(), BuildError> {
    let (line, tokens) = cursor.require(&expected.join(" "))?;
    if tokens.len() != expected.len()
        || tokens
            .iter()
            .zip(expected)
            .any(|(token, expected)| token != expected)
    {
        return Err(BuildError::Parse(format!(
            "line {line}: expected {expected:?}, got {tokens:?}"
        )));
    }
    Ok(())
}

struct AsciiCursor<R> {
    lines: std::io::Lines<BufReader<R>>,
    number: usize,
}

impl<R: std::io::Read> AsciiCursor<R> {
    fn new(reader: R) -> Self {
        Self {
            lines: BufReader::new(reader).lines(),
            number: 0,
        }
    }

    fn next_tokens(&mut self) -> Result<Option<(usize, Vec<String>)>, BuildError> {
        for line in self.lines.by_ref() {
            self.number += 1;
            let line = line.map_err(BuildError::Io)?;
            let tokens: Vec<String> = line.split_whitespace().map(ToString::to_string).collect();
            if !tokens.is_empty() {
                return Ok(Some((self.number, tokens)));
            }
        }
        Ok(None)
    }

    fn require(&mut self, what: &str) -> Result<(usize, Vec<String>), BuildError> {
        self.next_tokens()?
            .ok_or_else(|| BuildError::Parse(format!("unexpected EOF while expecting {what}")))
    }
}

#[cfg(test)]
mod tests {
    use super::parse_ascii_stl;
    use std::io::Write;

    #[test]
    fn solid_order_assigns_patch_ids_and_welds_vertices() {
        let mut stored = tempfile::NamedTempFile::new().unwrap();
        write!(
            stored,
            "\
solid back
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid back
solid front
facet normal 0 0 1
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
endsolid front
solid back
facet normal 0 0 1
outer loop
vertex 0 0 1
vertex 1 0 1
vertex 0 1 1
endloop
endfacet
endsolid back
"
        )
        .unwrap();
        let mut file = std::fs::File::open(stored.path()).unwrap();
        let parsed = parse_ascii_stl(&mut file).unwrap();
        assert_eq!(parsed.patch_names, ["back", "front"]);
        assert_eq!(parsed.anchor_to_patch_id, [0, 1, 0]);
        assert_eq!(parsed.vertices.len(), 6);
        assert_eq!(parsed.indices[0], parsed.indices[1]);
    }
}
