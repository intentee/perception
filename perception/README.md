# perception

Perceptual image similarity for Rust, built primarily around **SSIM** (the Structural Similarity Index). 

## Quick start

Compare two image files and read the similarity score:

```rust
use std::path::Path;

use perception::ImagePair;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let comparison = ImagePair::new(
        Path::new("original.png"),
        Path::new("distorted.png"),
    )
    .compare()?;

    // 1.0 means identical; lower means more structural difference.
    println!("similarity: {}", comparison.similarity());
    Ok(())
}
```

## Choosing a backend

`ImagePair` always uses the CPU backend. To select a backend explicitly — including CUDA —
use `Engine`:

```rust
use std::path::Path;

use perception::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = Engine::cpu();
    let comparison = engine.compare(
        Path::new("original.png"),
        Path::new("distorted.png"),
    )?;
    println!("similarity: {}", comparison.similarity());
    Ok(())
}
```

The CUDA backend requires the `cuda` feature and a working CUDA toolkit and GPU:

```toml
[dependencies]
perception = { version = "0.1", features = ["cuda"] }
```

```rust
# use std::path::Path;
use perception::Engine;

let engine = Engine::cuda()?; // creates the device context and compiles the kernels
let comparison = engine.compare(Path::new("original.png"), Path::new("distorted.png"))?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

| Feature | Default | Effect                                                       |
| ------- | ------- | ------------------------------------------------------------ |
| `cpu`   | yes     | Rayon-parallel CPU backend; enables `Engine::cpu()`.         |
| `cuda`  | no      | CUDA/GPU backend via `cudarc`; enables `Engine::cuda()`.     |

## Similarity map

Every comparison also yields a per-pixel `SimilarityMap`. Values are close to `1.0` where the
images match and lower where they differ. It can be written out as a grayscale PNG heatmap in
which differing regions appear brighter:

```rust
use std::path::Path;

use perception::ImagePair;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let map = ImagePair::new(
        Path::new("original.png"),
        Path::new("distorted.png"),
    )
    .compare()?
    .into_map();

    println!("{}x{} map", map.width(), map.height());
    let _values: &[f32] = map.values();
    map.write(Path::new("difference.png"))?;
    Ok(())
}
```

## Workspace layout

`perception` is a thin facade over a small family of crates:

| Crate                     | Role                                                        |
| ------------------------- | ----------------------------------------------------------- |
| `perception`              | High-level API (`ImagePair`, `Engine`, `Comparison`, …).    |
| `perception_metric`       | Backend-agnostic multi-scale SSIM metric engine.            |
| `perception_backend`      | Backend trait and shared types.                             |
| `perception_backend_cpu`  | CPU implementation of the backend.                          |
| `perception_backend_cuda` | CUDA/GPU implementation of the backend.                     |

## License

Licensed under the [Apache License, Version 2.0](https://github.com/intentee/perception/blob/main/LICENSE).
