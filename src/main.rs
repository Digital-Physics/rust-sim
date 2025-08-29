// These are "use" statements - similar to imports in other languages
// They bring external code into scope so we can use it
use rand::prelude::*;           // Import random number generation utilities
use std::sync::Arc;             // Arc = Atomically Reference Counted smart pointer (for shared ownership)
use std::time::{Duration, Instant}; // Time-related utilities
use wgpu::util::DeviceExt;      // WebGPU utilities for graphics
use winit::{                    // Window management library
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

// ======================== Simulation ========================

// #[derive(Clone)] is a "derive macro" that automatically generates
// a Clone implementation for this struct, allowing us to copy it
#[derive(Clone)]
struct Params {
    length: usize,      // usize is an unsigned integer type (size_t equivalent)
    num_types: usize,   // Grid dimensions and particle type count
    density: f32,       // f32 is a 32-bit floating point number
    radius: usize,
}

// Structs in Rust are like classes but without inheritance
// All fields are private by default unless marked 'pub'
struct ParticleGrid {
    params: Params,
    // Vec<T> is Rust's growable array type (like std::vector in C++)
    // Vec<Vec<u8>> creates a 2D vector - vector of vectors
    // u8 is an unsigned 8-bit integer (0-255)
    type_grid: Vec<Vec<u8>>,           // 2D grid storing particle types
    affinity: Vec<Vec<i8>>,            // i8 is signed 8-bit integer (-128 to 127)
    copy_type: Vec<u8>,                // Lookup tables for particle behavior
    replace_type: Vec<u8>,
    colors: Vec<[f32; 3]>,             // [f32; 3] is a fixed-size array of 3 floats (RGB)
    rng: ThreadRng,                    // Random number generator
}

// 'impl' blocks define methods for structs (like class methods)
impl ParticleGrid {
    // 'fn' declares a function
    // 'Self' refers to the struct type (ParticleGrid in this case)
    // This is an "associated function" (like a static method) since it doesn't take &self
    fn new_random() -> Self {
        // 'let mut' declares a mutable variable
        // 'let' alone would make it immutable (Rust's default)
        let mut rng = thread_rng();

        // Rust has powerful pattern matching and ranges
        // gen_range(20..=400) generates a number from 20 to 400 inclusive
        // The ..= syntax is an inclusive range
        let length = rng.gen_range(20..=400);
        let num_types = rng.gen_range(3..=15);
        
        // This creates a vector by transforming a range
        // .map() transforms each element, .collect() gathers results into a Vec
        let density_choices: Vec<f32> = (3..10).map(|i| i as f32 * 0.05).collect();
        // .choose() picks a random element, .unwrap_or() provides a default if None
        let density = *density_choices.choose(&mut rng).unwrap_or(&0.05);
        let radius = *[1usize, 2usize].choose(&mut rng).unwrap_or(&70);

        // Struct construction syntax - field names match the variables
        let params = Params {
            length,
            num_types,
            density,
            radius,
        };

        // vec! is a macro (note the !) that creates vectors
        // vec![value; count] creates a vector with 'count' copies of 'value'
        let mut type_grid = vec![vec![0u8; length]; length];
        
        // Nested loops with ranges
        // 0..length is an exclusive range (0 to length-1)
        for x in 0..length {
            for y in 0..length {
                // gen::<f32>() generates a random f32 between 0.0 and 1.0
                if rng.gen::<f32>() < density {
                    // Indexing with [] - Rust checks bounds at runtime in debug mode
                    type_grid[x][y] = rng.gen_range(1..=num_types as u8);
                } else {
                    type_grid[x][y] = 0;
                }
            }
        }

        // Initialize affinity matrix
        let mut affinity = vec![vec![0i8; num_types + 1]; num_types + 1];
        for t in 0..=num_types {  // ..= is inclusive range
            for u in 0..=num_types {
                // gen_bool(0.5) returns true 50% of the time
                let a = if rng.gen_bool(0.5) { 1 } else { 0 };
                // Conditional expression - Rust's equivalent of ternary operator
                affinity[t][u] = if a == 1 { 1 } else { -1 };
            }
        }

        // Initialize copy_type array
        let mut copy_type = vec![0u8; num_types + 1];
        for t in 0..=num_types {
            // This creates a vector by filtering a range
            // The |&c| syntax is a closure (lambda function)
            // &c pattern-matches and dereferences the parameter
            let mut choices: Vec<u8> = (1..=num_types as u8)
                .filter(|&c| c != t as u8)  // Keep only elements not equal to t
                .collect();
            // 'as' keyword casts between types
            copy_type[t] = *choices
                .choose(&mut rng)
                .unwrap_or(&((t as u8 % num_types as u8) + 1));
        }

        // Similar pattern for replace_type
        let mut replace_type = vec![0u8; num_types + 1];
        for t in 0..=num_types {
            let mut choices: Vec<u8> = (1..=num_types as u8)
                .filter(|&c| c != t as u8 && c != copy_type[t])
                .collect();
            // .is_empty() checks if vector has no elements
            if choices.is_empty() {
                // Fallback logic with conditional expression
                replace_type[t] = if num_types >= 1 { 1 } else { 0 };
            } else {
                // * dereferences the result since choose() returns Option<&T>
                replace_type[t] = *choices.choose(&mut rng).unwrap();
            }
        }

        // Initialize color palette
        // vec![value; count] syntax again
        let mut colors = vec![[0.1, 0.1, 0.1]; num_types + 1];
        for t in 1..=num_types {
            let h = (t as f32) / (num_types as f32);
            // Function call - hsv_to_rgb returns [f32; 3]
            colors[t] = hsv_to_rgb(h, 0.8, 1.0);
        }

        // Return the constructed struct
        // No 'return' keyword needed - last expression is automatically returned
        Self {
            params,
            type_grid,
            affinity,
            copy_type,
            replace_type,
            colors,
            rng,
        }
    }

    // #[inline] is an attribute suggesting the compiler should inline this function
    // &self means this method borrows the struct immutably (read-only access)
    #[inline]
    fn inside(&self, x: isize, y: isize) -> bool {
        // isize is signed integer type (can be negative)
        // Multiple conditions joined with &&
        // Cast with 'as' keyword
        x >= 0 && y >= 0 && (x as usize) < self.params.length && (y as usize) < self.params.length
    }

    // &mut self means this method borrows the struct mutably (can modify it)
    fn try_replace_particle(&mut self, x: usize, y: usize) {
        let p_type = self.type_grid[x][y];
        // Early return pattern - common in Rust
        if p_type == 0 {
            return;
        }
        let length = self.params.length;
        let ct = self.copy_type[p_type as usize];
        let rt = self.replace_type[p_type as usize];

        // Look for copy_type neighbor
        let mut has_copy_neighbor = false;
        // saturating_sub prevents underflow (returns 0 instead of panicking)
        // .min() clamps the upper bound
        for j in (y.saturating_sub(1))..=((y + 1).min(length - 1)) {
            for i in (x.saturating_sub(1))..=((x + 1).min(length - 1)) {
                if self.type_grid[i][j] == ct {
                    has_copy_neighbor = true;
                    break;
                }
            }
            if has_copy_neighbor {
                break;
            }
        }
        if !has_copy_neighbor {
            return;
        }

        // Replace all rt with ct in neighborhood
        for j in (y.saturating_sub(1))..=((y + 1).min(length - 1)) {
            for i in (x.saturating_sub(1))..=((x + 1).min(length - 1)) {
                if self.type_grid[i][j] == rt {
                    self.type_grid[i][j] = ct;
                }
            }
        }
    }

    // This method returns a tuple: (usize, usize)
    fn score_within_radius(&mut self, x: usize, y: usize) -> (usize, usize) {
        let length = self.params.length;
        let p_type = self.type_grid[x][y];
        
        // f32::EPSILON is the smallest representable positive f32
        let mut best: f32 = -1_000_000.0; // Underscores can separate digits for readability
        
        // Vec::new() creates an empty vector, vec! macro with initial values
        let mut tiebreak: Vec<(usize, usize)> = vec![(x, y)];

        // Nested loops to check adjacent cells
        for j in (y.saturating_sub(1))..=((y + 1).min(length - 1)) {
            for i in (x.saturating_sub(1))..=((x + 1).min(length - 1)) {
                if self.type_grid[i][j] != 0 {
                    continue; // Skip to next iteration
                }

                // i32 is 32-bit signed integer
                let mut score = 0i32;
                let mut cell_count = 0i32;

                // Calculate bounds for scoring region
                let rx0 = i.saturating_sub(self.params.radius);
                let rx1 = (i + self.params.radius).min(length - 1);
                let ry0 = j.saturating_sub(self.params.radius);
                let ry1 = (j + self.params.radius).min(length - 1);

                // Score calculation loop
                for yy in ry0..=ry1 {
                    for xx in rx0..=rx1 {
                        cell_count += 1;
                        let ct = self.type_grid[xx][yy];
                        if ct != 0 {
                            let a = self.affinity[p_type as usize][ct as usize];
                            // Conditional expression for scoring
                            score += if a == 1 { 1 } else { -1 };
                        }
                    }
                }

                // Normalize score and update best positions
                // .max() ensures we don't divide by zero
                let norm = score as f32 / (cell_count as f32).max(1.0);
                if norm > best {
                    best = norm;
                    tiebreak.clear(); // Remove all elements
                    tiebreak.push((i, j)); // Add new element
                } else if (norm - best).abs() < f32::EPSILON {
                    tiebreak.push((i, j));
                }
            }
        }

        // Return best position, with fallback
        if tiebreak.is_empty() {
            (x, y) // Return tuple
        } else {
            // * dereferences the Option<&T> from choose()
            *tiebreak.choose(&mut self.rng).unwrap()
        }
    }

    fn move_particle(&mut self, x: usize, y: usize) {
        let p_type = self.type_grid[x][y];
        if p_type == 0 {
            return;
        }
        // Destructure tuple into separate variables
        let (bx, by) = self.score_within_radius(x, y);
        if bx == x && by == y {
            // No movement needed
            return;
        }
        // Update grid state
        self.type_grid[bx][by] = p_type;
        self.type_grid[x][y] = 0;
    }

    /// Triple slash creates documentation comments (like Javadoc)
    /// These appear in generated documentation
    fn step(&mut self) {
        let length = self.params.length;
        let total_cells = (length * length) as f32;
        // .floor() rounds down, 'as usize' casts to integer
        let updates = (0.2 * self.params.density * total_cells).floor() as usize;

        if updates == 0 {
            return;
        }

        // Collect current non-empty cells
        // Vec::new() creates empty vector, reserve() pre-allocates capacity
        let mut particles: Vec<(usize, usize)> = Vec::new();
        particles.reserve(length * length / 2);
        
        for x in 0..length {
            for y in 0..length {
                if self.type_grid[x][y] != 0 {
                    particles.push((x, y));
                }
            }
        }

        if particles.is_empty() {
            return;
        }

        // Update random particles
        for _ in 0..updates {
            if particles.is_empty() {
                break;
            }
            let idx = self.rng.gen_range(0..particles.len());
            let (x, y) = particles[idx];
            
            // Check if particle still exists (might have moved)
            if self.type_grid[x][y] == 0 {
                // swap_remove is O(1) - moves last element to this position
                particles.swap_remove(idx);
                continue;
            }
            self.try_replace_particle(x, y);
            self.move_particle(x, y);
        }
    }
}

// Free function (not associated with any struct)
// Takes parameters by value and returns by value
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    // .fract() gets fractional part, .floor() gets integer part
    let h6 = (h * 6.0).fract();
    let i = (h * 6.0).floor() as i32 % 6;
    let f = h6;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    
    // match is Rust's pattern matching - like switch but more powerful
    // Each arm must return the same type
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q], // _ is catch-all pattern
    }
}

// ======================== Rendering ========================

// These are derive macros that automatically implement traits
// Copy: can be copied bit-for-bit (like memcpy)
// Clone: can be explicitly copied
// Debug: can be formatted with {:?}
// Pod: "Plain Old Data" - safe to cast to/from bytes
// Zeroable: safe to zero out the memory
#[repr(C)] // Use C memory layout (important for GPU)
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3], // 3D position
    color: [f32; 3],    // RGB color
}

impl Vertex {
    // Associated function that returns vertex buffer layout description
    // Lifetime parameter 'a indicates how long the returned reference lives
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            // size_of::<T>() gets size in bytes of type T
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            // &[...] creates a slice (view into an array/vector)
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// Main rendering state struct
struct State {
    // These are wgpu types for GPU rendering
    surface: wgpu::Surface<'static>,     // 'static lifetime means lives for entire program
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>, // Window size
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32, // u32 is 32-bit unsigned integer

    grid: ParticleGrid, // Our simulation
    last_update: Instant, // For timing updates
}

impl State {
    // async/await for asynchronous programming
    // Arc<Window> is atomic reference counting - allows shared ownership
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        // GPU initialization code
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        // .clone() creates another Arc pointing to same Window
        let surface = instance.create_surface(window.clone()).unwrap();

        // .await suspends function until future completes
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface), // Some() wraps value in Option
                force_fallback_adapter: false,
            })
            .await
            // .expect() panics with message if Result is Err
            .expect("No suitable GPU adapters found on the system!");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None, // None is empty Option
                },
                None,
            )
            .await
            .expect("Failed to create device");

        // Surface configuration
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter() // Create iterator
            .copied() // Copy each element (since iter gives references)
            .find(|f| f.is_srgb()) // Find first matching element
            .unwrap_or(surface_caps.formats[0]); // Default if none found

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![], // Empty vector
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Raw string literal - no escape sequences processed
        let shader_source = r#"
struct VertexIn {
  @location(0) position: vec3<f32>,
  @location(1) color: vec3<f32>,
};

struct VertexOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
  var out: VertexOut;
  out.pos = vec4<f32>(v.position, 1.0);
  out.color = v.color;
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
"#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"), // Some() wraps string in Option
            // .into() converts string to required type
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[], // Empty slice
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()], // Call associated function
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0, // Bitwise NOT of 0 (all bits set)
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let grid = ParticleGrid::new_random();

        // println! is a macro for formatted printing
        println!("Initial grid state ({}x{}, {} types, dens {:.2}, r={}):",
            grid.params.length, grid.params.length, grid.params.num_types, grid.params.density, grid.params.radius
        );
        // Debug output of grid state
        for y in 0..grid.params.length {
            for x in 0..grid.params.length {
                print!("{}", grid.type_grid[x][y]);
            }
            println!(); // Print newline
        }

        // Destructure tuple returned by function
        let (vertices, indices) = Self::create_grid_mesh(&grid);

        // Buffer allocation
        let max_vertices = (grid.params.length * grid.params.length * 4).max(vertices.len());
        let vb_size = (max_vertices * std::mem::size_of::<Vertex>()) as u64;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: vb_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if !vertices.is_empty() {
            // bytemuck::cast_slice converts Vec<Vertex> to &[u8] safely
            queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }

        // create_buffer_init is a convenience method
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let num_indices = indices.len() as u32;

        // Return constructed State
        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            grid,
            last_update: Instant::now(),
        }
    }

    // Static method that returns a tuple of vectors
    fn create_grid_mesh(grid: &ParticleGrid) -> (Vec<Vertex>, Vec<u16>) {
        // Vec::new() creates empty vectors
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let l = grid.params.length as f32;
        let cell = 2.0 / l; // Cell size in normalized coordinates
        let pad = 0.98; // Padding factor
        
        for x in 0..grid.params.length {
            for y in 0..grid.params.length {
                let t = grid.type_grid[x][y];
                if t == 0 {
                    continue; // Skip empty cells
                }
                let color = grid.colors[t as usize];

                // Convert grid coordinates to normalized device coordinates
                let x1 = -1.0 + (x as f32) * cell;
                let y1 = 1.0 - (y as f32) * cell;
                let x2 = x1 + cell * pad;
                let y2 = y1 - cell * pad;

                // Current vertex count for indexing
                let base = vertices.len() as u16;
                
                // .extend_from_slice() appends slice to vector
                vertices.extend_from_slice(&[
                    Vertex {
                        position: [x1, y1, 0.0],
                        color,
                    },
                    Vertex {
                        position: [x2, y1, 0.0],
                        color,
                    },
                    Vertex {
                        position: [x2, y2, 0.0],
                        color,
                    },
                    Vertex {
                        position: [x1, y2, 0.0],
                        color,
                    },
                ]);

                // Two triangles per quad
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }

        println!(
            "Created mesh with {} vertices and {} indices for ~{} non-empty cells",
            vertices.len(),
            indices.len(),
            vertices.len() / 4
        );
        (vertices, indices) // Return tuple
    }

    fn update(&mut self) {
        // Check if enough time has elapsed for update
        if self.last_update.elapsed() >= Duration::from_millis(100) {
            self.grid.step(); // Update simulation
            self.last_update = Instant::now();

            // Rebuild mesh since particle positions changed
            let (vertices, indices) = Self::create_grid_mesh(&self.grid);

            if !vertices.is_empty() {
                // Update vertex buffer with new data
                self.queue
                    .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
                    
                // Recreate index buffer (size might have changed)
                self.index_buffer = self
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Index Buffer (dynamic)"),
                        contents: bytemuck::cast_slice(&indices),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                self.num_indices = indices.len() as u32;
            } else {
                self.num_indices = 0;
            }
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    // Result<T, E> is Rust's error handling type
    // Ok(value) for success, Err(error) for failure
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // ? operator propagates errors up the call stack
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

        // Scope braces create a new scope - pass will be dropped at end
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            // Set up rendering pipeline and buffers
            pass.set_pipeline(&self.render_pipeline);
            // .. creates a slice of the entire buffer
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            // Only draw if we have indices to draw
            if self.num_indices > 0 {
                // Draw indexed triangles
                // 0..self.num_indices is range of indices to draw
                // 0 is vertex offset, 0..1 is instance range
                pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
        } // render pass automatically ends here when 'pass' goes out of scope

        // Submit command buffer to GPU queue
        // std::iter::once creates an iterator with one element
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present(); // Present frame to screen
        Ok(()) // Return Ok with unit type ()
    }
}

// ======================== App / Winit ========================

// Application state struct
struct App {
    // Option<T> represents optional values - Some(T) or None
    state: Option<State>,
    window: Option<Arc<Window>>,
}

// Implement the ApplicationHandler trait for our App
// Traits are like interfaces in other languages
impl ApplicationHandler for App {
    // This method is called when the application should create its window
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Arc::new creates a new atomically reference-counted pointer
        let window = Arc::new(
            event_loop
                .create_window(
                    // Method chaining - each method returns modified builder
                    Window::default_attributes()
                        .with_title("Particle Affinity (wgpu)")
                        .with_inner_size(winit::dpi::LogicalSize::new(900, 900)),
                )
                .unwrap(), // Panic if window creation fails
        );

        // pollster::block_on runs async code synchronously
        // This blocks the current thread until the async function completes
        let state = pollster::block_on(State::new(window.clone()));
        self.window = Some(window.clone());
        self.state = Some(state);

        // Request initial redraw
        // if let Some(w) = &self.window - pattern matching on Option
        // Only executes if window is Some(value), not None
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    // Handle window events
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId, // _ prefix means unused parameter
        event: WindowEvent,
    ) {
        // match on the event type
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit(); // Quit the application
            }
            WindowEvent::Resized(size) => {
                // Handle window resize
                if let Some(state) = &mut self.state {
                    state.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                // Handle redraw request
                if let Some(state) = &mut self.state {
                    state.update(); // Update simulation
                    // Handle different render results
                    match state.render() {
                        Ok(_) => {} // Success - do nothing
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        // Other errors - print debug info
                        Err(e) => eprintln!("{e:?}"), // eprintln! prints to stderr
                    }
                }
                // Request next frame
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {} // Ignore all other events
        }
    }

    // Called when event loop is about to wait for events
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

// Main function - entry point of the program
fn main() {
    // Initialize logging (for debug output)
    env_logger::init();
    
    // Create event loop - this manages the window and events
    let event_loop = EventLoop::new().unwrap();
    // Set to Poll mode for continuous updates (vs Wait which only updates on events)
    event_loop.set_control_flow(ControlFlow::Poll);

    // Create application instance
    let mut app = App {
        state: None,
        window: None,
    };

    // Run the application - this blocks until app exits
    // &mut creates a mutable reference
    event_loop.run_app(&mut app).unwrap();
}