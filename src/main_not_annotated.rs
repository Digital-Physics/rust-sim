use rand::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

// ======================== Simulation ========================

#[derive(Clone)]
struct Params {
    length: usize,
    num_types: usize,
    density: f32, // fraction of non-zero cells overall
    radius: usize,
}

struct ParticleGrid {
    params: Params,
    // Python indexes as type_grid[x][y]; we’ll mirror that for logic parity.
    type_grid: Vec<Vec<u8>>,           // size length x length, entries in [0..=num_types]
    affinity: Vec<Vec<i8>>,            // [type][type] => +1 (attract), -1 (repel)
    copy_type: Vec<u8>,                // for each type t, copy_type[t] != 0 and != t
    replace_type: Vec<u8>,             // for each type t, replace_type[t] != t and != copy_type[t]
    colors: Vec<[f32; 3]>,             // color palette: index 0 is background
    rng: ThreadRng,
}

impl ParticleGrid {
    fn new_random() -> Self {
        let mut rng = thread_rng();

        // mirror Python create(): random length, num_types, density, radius
        let length = rng.gen_range(20..=400);
        let num_types = rng.gen_range(3..=15);
        let density_choices: Vec<f32> = (3..10).map(|i| i as f32 * 0.05).collect(); // 0.15..0.45
        let density = *density_choices.choose(&mut rng).unwrap_or(&0.3);
        let radius = *[1usize, 2usize].choose(&mut rng).unwrap_or(&1);

        let params = Params {
            length,
            num_types,
            density,
            radius,
        };

        // type_grid: for each cell choose 0 (empty) with 1-density, else uniform among 1..=num_types
        let mut type_grid = vec![vec![0u8; length]; length];
        for x in 0..length {
            for y in 0..length {
                if rng.gen::<f32>() < density {
                    type_grid[x][y] = rng.gen_range(1..=num_types as u8);
                } else {
                    type_grid[x][y] = 0;
                }
            }
        }

        // affinity: for each type (0..num_types), choose 0 or 1 with 0.5 each;
        // Python treats 1 => +1, 0 => -1 in score accumulation.
        let mut affinity = vec![vec![0i8; num_types + 1]; num_types + 1];
        for t in 0..=num_types {
            for u in 0..=num_types {
                let a = if rng.gen_bool(0.5) { 1 } else { 0 };
                affinity[t][u] = if a == 1 { 1 } else { -1 };
            }
        }

        // copy_type[t] for t in 0..=num_types; Python builds for all indices, but constraints apply for t>=1.
        // Enforce: copy_type[t] != t and != 0. For t=0, choose any valid (we won’t use it).
        let mut copy_type = vec![0u8; num_types + 1];
        for t in 0..=num_types {
            let mut choices: Vec<u8> = (1..=num_types as u8)
                .filter(|&c| c != t as u8)
                .collect();
            copy_type[t] = *choices
                .choose(&mut rng)
                .unwrap_or(&((t as u8 % num_types as u8) + 1));
        }

        // replace_type[t]: in 1..=num_types, not equal to t, and not equal to copy_type[t]
        let mut replace_type = vec![0u8; num_types + 1];
        for t in 0..=num_types {
            let mut choices: Vec<u8> = (1..=num_types as u8)
                .filter(|&c| c != t as u8 && c != copy_type[t])
                .collect();
            if choices.is_empty() {
                // Fallback if num_types == 1 edge case (shouldn’t happen here)
                replace_type[t] = if num_types >= 1 { 1 } else { 0 };
            } else {
                replace_type[t] = *choices.choose(&mut rng).unwrap();
            }
        }

        // colors: background + num_types hues
        let mut colors = vec![[0.1, 0.1, 0.1]; num_types + 1]; // 0 = dark gray background
        for t in 1..=num_types {
            let h = (t as f32) / (num_types as f32); // 0..1
            colors[t] = hsv_to_rgb(h, 0.8, 1.0);
        }

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

    #[inline]
    fn inside(&self, x: isize, y: isize) -> bool {
        x >= 0 && y >= 0 && (x as usize) < self.params.length && (y as usize) < self.params.length
    }

    fn try_replace_particle(&mut self, x: usize, y: usize) {
        let p_type = self.type_grid[x][y];
        if p_type == 0 {
            return;
        }
        let length = self.params.length;
        let ct = self.copy_type[p_type as usize];
        let rt = self.replace_type[p_type as usize];

        // Python logic: if any neighbor equals copy_type[p_type], then in neighborhood replace all rt with ct.
        let mut has_copy_neighbor = false;
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

        for j in (y.saturating_sub(1))..=((y + 1).min(length - 1)) {
            for i in (x.saturating_sub(1))..=((x + 1).min(length - 1)) {
                if self.type_grid[i][j] == rt {
                    self.type_grid[i][j] = ct;
                }
            }
        }
    }

    fn score_within_radius(&mut self, x: usize, y: usize) -> (usize, usize) {
        let length = self.params.length;
        let p_type = self.type_grid[x][y];
        // default: stay put
        let mut best: f32 = -1_000_000.0;
        let mut tiebreak: Vec<(usize, usize)> = vec![(x, y)];

        // For each open adjacent cell, compute normalized score across a (2r+1)x(2r+1) box centered at that cell.
        for j in (y.saturating_sub(1))..=((y + 1).min(length - 1)) {
            for i in (x.saturating_sub(1))..=((x + 1).min(length - 1)) {
                if self.type_grid[i][j] != 0 {
                    continue; // only open cells are considered
                }

                let mut score = 0i32;
                let mut cell_count = 0i32;

                let rx0 = i.saturating_sub(self.params.radius);
                let rx1 = (i + self.params.radius).min(length - 1);
                let ry0 = j.saturating_sub(self.params.radius);
                let ry1 = (j + self.params.radius).min(length - 1);

                for yy in ry0..=ry1 {
                    for xx in rx0..=rx1 {
                        cell_count += 1;
                        let ct = self.type_grid[xx][yy];
                        if ct != 0 {
                            // +1 if attracted, -1 if not
                            let a = self.affinity[p_type as usize][ct as usize];
                            score += if a == 1 { 1 } else { -1 };
                        }
                    }
                }

                let norm = score as f32 / (cell_count as f32).max(1.0);
                if norm > best {
                    best = norm;
                    tiebreak.clear();
                    tiebreak.push((i, j));
                } else if (norm - best).abs() < f32::EPSILON {
                    tiebreak.push((i, j));
                }
            }
        }

        if tiebreak.is_empty() {
            (x, y)
        } else {
            *tiebreak.choose(&mut self.rng).unwrap()
        }
    }

    fn move_particle(&mut self, x: usize, y: usize) {
        let p_type = self.type_grid[x][y];
        if p_type == 0 {
            return;
        }
        let (bx, by) = self.score_within_radius(x, y);
        if bx == x && by == y {
            // stayed
            return;
        }
        // move
        self.type_grid[bx][by] = p_type;
        self.type_grid[x][y] = 0;
    }

    /// Asynchronous step: update ~20% * density * length^2 random particles, like Python.
    fn step(&mut self) {
        let length = self.params.length;
        let total_cells = (length * length) as f32;
        let updates = (0.2 * self.params.density * total_cells).floor() as usize;

        if updates == 0 {
            return;
        }

        // collect current non-empty cells
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

        for _ in 0..updates {
            if particles.is_empty() {
                break;
            }
            let idx = self.rng.gen_range(0..particles.len());
            let (x, y) = particles[idx];
            // particle might have moved/cleared by earlier iterations; re-check
            if self.type_grid[x][y] == 0 {
                // remove stale entry
                particles.swap_remove(idx);
                continue;
            }
            self.try_replace_particle(x, y);
            self.move_particle(x, y);
        }
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    // h in [0,1], s in [0,1], v in [0,1]
    let h6 = (h * 6.0).fract();
    let i = (h * 6.0).floor() as i32 % 6;
    let f = h6;
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

// ======================== Rendering ========================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}
impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
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

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    grid: ParticleGrid,
    last_update: Instant,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            dx12_shader_compiler: Default::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapters found on the system!");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

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
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
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
                cull_mode: None, // disable culling to avoid winding issues
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let grid = ParticleGrid::new_random();

        // Optional: print initial grid for debugging (like your log)
        println!("Initial grid state ({}x{}, {} types, dens {:.2}, r={}):",
            grid.params.length, grid.params.length, grid.params.num_types, grid.params.density, grid.params.radius
        );
        for y in 0..grid.params.length {
            for x in 0..grid.params.length {
                print!("{}", grid.type_grid[x][y]);
            }
            println!();
        }

        let (vertices, indices) = Self::create_grid_mesh(&grid);

        // allocate a reasonably large vertex buffer up-front; max quads = L*L, 4 verts per quad
        let max_vertices = (grid.params.length * grid.params.length * 4).max(vertices.len());
        let vb_size = (max_vertices * std::mem::size_of::<Vertex>()) as u64;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vertex Buffer"),
            size: vb_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if !vertices.is_empty() {
            queue.write_buffer(&vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        }

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        });

        let num_indices = indices.len() as u32;

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

    fn create_grid_mesh(grid: &ParticleGrid) -> (Vec<Vertex>, Vec<u16>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let l = grid.params.length as f32;
        let cell = 2.0 / l; // full grid fills [-1, 1] x [-1, 1]
        let pad = 0.98; // slight gap between cells
        for x in 0..grid.params.length {
            for y in 0..grid.params.length {
                let t = grid.type_grid[x][y];
                if t == 0 {
                    continue;
                }
                let color = grid.colors[t as usize];

                // map (x,y) to NDC; Python had (x,y) as (col,row). y grows downward; NDC y grows upward.
                let x1 = -1.0 + (x as f32) * cell;
                let y1 = 1.0 - (y as f32) * cell;
                let x2 = x1 + cell * pad;
                let y2 = y1 - cell * pad;

                let base = vertices.len() as u16;
                vertices.extend_from_slice(&[
                    Vertex {
                        position: [x1, y1, 0.0],
                        color,
                    }, // top-left
                    Vertex {
                        position: [x2, y1, 0.0],
                        color,
                    }, // top-right
                    Vertex {
                        position: [x2, y2, 0.0],
                        color,
                    }, // bottom-right
                    Vertex {
                        position: [x1, y2, 0.0],
                        color,
                    }, // bottom-left
                ]);

                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }

        println!(
            "Created mesh with {} vertices and {} indices for ~{} non-empty cells",
            vertices.len(),
            indices.len(),
            vertices.len() / 4
        );
        (vertices, indices)
    }

    fn update(&mut self) {
        // drive the simulation at ~10 Hz (100 ms)
        if self.last_update.elapsed() >= Duration::from_millis(100) {
            self.grid.step();
            self.last_update = Instant::now();

            // rebuild mesh (counts change as particles move)
            let (vertices, indices) = Self::create_grid_mesh(&self.grid);

            if !vertices.is_empty() {
                self.queue
                    .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
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
            // no camera/projection; nothing else to do
        }
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

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

            pass.set_pipeline(&self.render_pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            if self.num_indices > 0 {
                pass.draw_indexed(0..self.num_indices, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        Ok(())
    }
}

// ======================== App / Winit ========================

struct App {
    state: Option<State>,
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Particle Affinity (wgpu)")
                        .with_inner_size(winit::dpi::LogicalSize::new(900, 900)),
                )
                .unwrap(),
        );

        let state = pollster::block_on(State::new(window.clone()));
        self.window = Some(window.clone());
        self.state = Some(state);

        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    state.resize(size);
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(state) = &mut self.state {
                    state.update();
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                        Err(e) => eprintln!("{e:?}"),
                    }
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        state: None,
        window: None,
    };

    event_loop.run_app(&mut app).unwrap();
}
