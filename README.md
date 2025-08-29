This is my first Rust repository. It has 'particle affinity model' code that was ported over from python/js.

I'm trying to replicate the results in this video:
https://www.youtube.com/watch?v=xqdVHXkGCAw

I think to get the more interesting results shown at the end, we need to increase the radius size which would allow each particle to see more of the environment... we should also think about the density of particles... and making the simulation representation more efficient.

Some Rust commands:
- `cargo new <project_name>` - Creates a new Rust project with directory structure and Cargo.toml (like package.json in JS or requirements.txt in Python)
- `cargo build` - Compiles your project and creates an executable in target/debug/
- `cargo run` - Compiles and runs your project in one command
- `cargo clean` - Removes build artifacts from the target/ directory

Note: No virtual environment management needed (Rust handles dependencies differently than Python)
Note: There is still some bugs in the window visualization because sometimes, depending on the random dimension, some of the window is black.

