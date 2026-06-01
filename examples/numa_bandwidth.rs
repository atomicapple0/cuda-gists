//! NUMA within-domain vs cross-domain GPU<->host bandwidth via cudaMemcpyAsync.
//!
//! Compares two host allocation strategies at saturation (1 GiB buffers, all GPUs):
//!   1. libnuma `numa_alloc_onnode` + `cuMemHostRegister`
//!   2. `cuMemHostAlloc` on a GPU context (GPUs 0..3 = domain A, 4..7 = domain B)
//!
//! Mirrors dkv/hack/numa-gpu-host-bench. Run with:
//!   cargo run --release --example numa_bandwidth

use cuda_gists::numa::{self, NumaHostBuffer};
use cuda_gists::*;

const GB: usize = 1024 * 1024 * 1024;
const BUFFER_SIZE: usize = GB;
const WARMUP_ITERS: usize = 3;
const TIMED_ITERS: usize = 10;
const COPIES_PER_GPU: &[usize] = &[1, 2, 4];
const NUM_DEVICES: usize = 8;
const GPUS_PER_DOMAIN: usize = 4;

#[derive(Clone, Copy)]
enum Direction {
    H2D,
    D2H,
}

struct BenchResult {
    copies_per_gpu: usize,
    within_gbps: f64,
    cross_gbps: f64,
}

struct TransferSetup {
    gpu_bufs: Vec<Vec<Buffer>>,
    host_bufs: Vec<Vec<Buffer>>,
    numa_hosts: Vec<Vec<NumaHostBuffer>>,
    pinned_alloc_devs: Vec<Vec<usize>>,
}

impl TransferSetup {
    fn free(mut self, streams: &[Stream]) {
        for device_id in 0..self.gpu_bufs.len() {
            for gpu_buf in &self.gpu_bufs[device_id] {
                streams[device_id].free_buffer_sync(gpu_buf);
            }
            if !self.numa_hosts[device_id].is_empty() {
                for host_buf in self.numa_hosts[device_id].drain(..) {
                    host_buf.free();
                }
            } else {
                for (host_buf, alloc_dev) in self.host_bufs[device_id]
                    .drain(..)
                    .zip(self.pinned_alloc_devs[device_id].drain(..))
                {
                    streams[alloc_dev].free_buffer_sync(&host_buf);
                }
            }
        }
        for stream in streams {
            stream.synchronize();
        }
    }
}

fn bandwidth_gb_s(elapsed: std::time::Duration, total_bytes: usize) -> f64 {
    total_bytes as f64 / elapsed.as_secs_f64() / 1e9
}

fn pinned_alloc_device(device_id: usize, cross_domain: bool) -> usize {
    if !cross_domain {
        device_id
    } else if device_id < GPUS_PER_DOMAIN {
        device_id + GPUS_PER_DOMAIN
    } else {
        device_id - GPUS_PER_DOMAIN
    }
}

fn setup_libnuma(
    streams: &[Stream],
    gpu_numa_nodes: &[i32],
    node_a: i32,
    node_b: i32,
    cross_domain: bool,
    copies_per_gpu: usize,
) -> TransferSetup {
    let mut gpu_bufs = Vec::with_capacity(streams.len());
    let mut host_bufs = Vec::with_capacity(streams.len());
    let mut numa_hosts = Vec::with_capacity(streams.len());
    let mut pinned_alloc_devs = Vec::with_capacity(streams.len());

    for device_id in 0..streams.len() {
        let gpu_node = gpu_numa_nodes[device_id];
        let host_node = if cross_domain {
            numa::other_node(gpu_node, node_a, node_b)
        } else {
            gpu_node
        };

        let mut hosts = Vec::with_capacity(copies_per_gpu);
        let mut host_views = Vec::with_capacity(copies_per_gpu);
        let mut gpus = Vec::with_capacity(copies_per_gpu);
        for _ in 0..copies_per_gpu {
            let host = NumaHostBuffer::alloc_onnode(BUFFER_SIZE, host_node);
            host_views.push(host.as_buffer(&streams[device_id].ctx));
            hosts.push(host);
            gpus.push(streams[device_id].create_buffer_async(BUFFER_SIZE, AddressSpace::Device));
        }
        numa_hosts.push(hosts);
        host_bufs.push(host_views);
        gpu_bufs.push(gpus);
        pinned_alloc_devs.push(Vec::new());
    }

    TransferSetup {
        gpu_bufs,
        host_bufs,
        numa_hosts,
        pinned_alloc_devs,
    }
}

fn setup_cuda_pinned(
    streams: &[Stream],
    cross_domain: bool,
    copies_per_gpu: usize,
) -> TransferSetup {
    let mut gpu_bufs = Vec::with_capacity(streams.len());
    let mut host_bufs = Vec::with_capacity(streams.len());
    let mut numa_hosts = Vec::with_capacity(streams.len());
    let mut pinned_alloc_devs = Vec::with_capacity(streams.len());

    for device_id in 0..streams.len() {
        let alloc_dev = pinned_alloc_device(device_id, cross_domain);
        let mut hosts = Vec::with_capacity(copies_per_gpu);
        let mut allocs = Vec::with_capacity(copies_per_gpu);
        let mut gpus = Vec::with_capacity(copies_per_gpu);
        for _ in 0..copies_per_gpu {
            hosts.push(streams[alloc_dev].create_buffer_async(BUFFER_SIZE, AddressSpace::Pinned));
            allocs.push(alloc_dev);
            gpus.push(streams[device_id].create_buffer_async(BUFFER_SIZE, AddressSpace::Device));
        }
        host_bufs.push(hosts);
        gpu_bufs.push(gpus);
        numa_hosts.push(Vec::new());
        pinned_alloc_devs.push(allocs);
    }

    TransferSetup {
        gpu_bufs,
        host_bufs,
        numa_hosts,
        pinned_alloc_devs,
    }
}

fn run_round(
    streams: &[Stream],
    gpu_bufs: &[Vec<Buffer>],
    host_bufs: &[Vec<Buffer>],
    dir: Direction,
) {
    for device_id in 0..gpu_bufs.len() {
        for (gpu_buf, host_buf) in gpu_bufs[device_id].iter().zip(host_bufs[device_id].iter()) {
            match dir {
                Direction::H2D => streams[device_id].memcpy_async(gpu_buf, host_buf),
                Direction::D2H => streams[device_id].memcpy_async(host_buf, gpu_buf),
            }
        }
    }
}

fn benchmark_transfer(
    streams: &[Stream],
    setup: TransferSetup,
    copies_per_gpu: usize,
    dir: Direction,
) -> f64 {
    let num_devices = streams.len();
    for stream in streams {
        stream.synchronize();
    }

    for _ in 0..WARMUP_ITERS {
        run_round(streams, &setup.gpu_bufs, &setup.host_bufs, dir);
    }
    for stream in streams {
        stream.synchronize();
    }

    let t0 = std::time::Instant::now();
    for _ in 0..TIMED_ITERS {
        run_round(streams, &setup.gpu_bufs, &setup.host_bufs, dir);
    }
    for stream in streams {
        stream.synchronize();
    }

    let total_bytes = BUFFER_SIZE * copies_per_gpu * TIMED_ITERS * num_devices;
    let bw = bandwidth_gb_s(t0.elapsed(), total_bytes);
    setup.free(streams);
    bw
}

fn run_scenario<F>(
    streams: &[Stream],
    copies_per_gpu: usize,
    dir: Direction,
    label: &str,
    mut setup_fn: F,
) -> (f64, f64)
where
    F: FnMut(bool) -> TransferSetup,
{
    let total = copies_per_gpu * streams.len();
    let dir_label = match dir {
        Direction::H2D => "H2D",
        Direction::D2H => "D2H",
    };
    log!(
        "Running {label} {dir_label} within-domain copies/gpu={copies_per_gpu} (total {total}) ..."
    );
    let within = benchmark_transfer(streams, setup_fn(false), copies_per_gpu, dir);
    log!("    aggregate {within:.1} GB/s");

    log!(
        "Running {label} {dir_label} cross-domain  copies/gpu={copies_per_gpu} (total {total}) ..."
    );
    let cross = benchmark_transfer(streams, setup_fn(true), copies_per_gpu, dir);
    log!("    aggregate {cross:.1} GB/s");
    (within, cross)
}

fn print_table(title: &str, num_devices: usize, results: &[BenchResult]) {
    log!("");
    log!("Aggregate {title} (buffer={} GiB)", BUFFER_SIZE / GB);
    log!("copies/gpu | total |  within-domain |   cross-domain | cross/within");
    log!("-------------------------------------------------------------------");
    for r in results {
        let total = r.copies_per_gpu * num_devices;
        let ratio = if r.within_gbps > 0.0 {
            format!("{:>11.2}x", r.cross_gbps / r.within_gbps)
        } else {
            "        n/a".to_string()
        };
        log!(
            "{:>10} | {:>5} | {:>9.1} GB/s | {:>9.1} GB/s | {ratio}",
            r.copies_per_gpu,
            total,
            r.within_gbps,
            r.cross_gbps,
        );
    }
}

fn run_libnuma_benchmarks(streams: &[Stream], gpu_numa_nodes: &[i32], node_a: i32, node_b: i32) {
    log!("libnuma: numa_alloc_onnode + cuMemHostRegister");
    log!("Domain A = NUMA node {node_a}; Domain B = NUMA node {node_b}");

    let mut h2d = Vec::new();
    let mut d2h = Vec::new();
    for &copies_per_gpu in COPIES_PER_GPU {
        let nodes = gpu_numa_nodes.to_vec();
        let (within, cross) = run_scenario(
            streams,
            copies_per_gpu,
            Direction::H2D,
            "libnuma",
            |cross| setup_libnuma(streams, &nodes, node_a, node_b, cross, copies_per_gpu),
        );
        h2d.push(BenchResult {
            copies_per_gpu,
            within_gbps: within,
            cross_gbps: cross,
        });

        let nodes = gpu_numa_nodes.to_vec();
        let (within, cross) = run_scenario(
            streams,
            copies_per_gpu,
            Direction::D2H,
            "libnuma",
            |cross| setup_libnuma(streams, &nodes, node_a, node_b, cross, copies_per_gpu),
        );
        d2h.push(BenchResult {
            copies_per_gpu,
            within_gbps: within,
            cross_gbps: cross,
        });
    }
    print_table("H2D via libnuma", streams.len(), &h2d);
    print_table("D2H via libnuma", streams.len(), &d2h);
}

fn run_cuda_pinned_benchmarks(streams: &[Stream]) {
    log!("");
    log!(
        "cuMemHostAlloc (flags=0): GPUs 0..{} = domain A, {}..{} = domain B",
        GPUS_PER_DOMAIN - 1,
        GPUS_PER_DOMAIN,
        streams.len() - 1
    );
    log!("within = alloc on copying GPU; cross = alloc on paired GPU in other domain");

    let mut h2d = Vec::new();
    let mut d2h = Vec::new();
    for &copies_per_gpu in COPIES_PER_GPU {
        let (within, cross) = run_scenario(
            streams,
            copies_per_gpu,
            Direction::H2D,
            "cuMemHostAlloc",
            |cross| setup_cuda_pinned(streams, cross, copies_per_gpu),
        );
        h2d.push(BenchResult {
            copies_per_gpu,
            within_gbps: within,
            cross_gbps: cross,
        });

        let (within, cross) = run_scenario(
            streams,
            copies_per_gpu,
            Direction::D2H,
            "cuMemHostAlloc",
            |cross| setup_cuda_pinned(streams, cross, copies_per_gpu),
        );
        d2h.push(BenchResult {
            copies_per_gpu,
            within_gbps: within,
            cross_gbps: cross,
        });
    }
    print_table("H2D via cuMemHostAlloc", streams.len(), &h2d);
    print_table("D2H via cuMemHostAlloc", streams.len(), &d2h);
}

fn gpu_numa_nodes(info: &[numa::GpuNumaInfo]) -> Vec<i32> {
    (0..NUM_DEVICES)
        .map(|i| {
            info.iter()
                .find(|g| g.device_id == i as i32)
                .map(|g| g.numa_node)
                .unwrap_or(-1)
        })
        .collect()
}

fn main() {
    log!("NUMA GPU<->host bandwidth benchmark");

    let ctxs: Vec<Context> = (0..NUM_DEVICES).map(|i| Context::new(i as i32)).collect();
    let streams: Vec<Stream> = ctxs.iter().map(|ctx| ctx.create_stream()).collect();

    log!("Enabling peer access");
    enable_peer_access(&ctxs);
    for stream in &streams {
        stream.synchronize();
    }

    let gpu_numa_info =
        numa::discover_gpu_numa_nodes().expect("failed to discover GPU NUMA topology");
    log!("GPU NUMA topology:");
    for g in &gpu_numa_info {
        log!("  GPU {}: numa_node={}", g.device_id, g.numa_node);
    }
    let nodes = gpu_numa_nodes(&gpu_numa_info);

    log!("Bandwidth in decimal GB/s (10^9 B/s)");
    log!(
        "buffer={} GiB, warmup={}, timed={}",
        BUFFER_SIZE / GB,
        WARMUP_ITERS,
        TIMED_ITERS
    );

    if let Some((node_a, node_b)) = numa::two_gpu_numa_domains(&gpu_numa_info) {
        run_libnuma_benchmarks(&streams, &nodes, node_a, node_b);
    } else {
        log!("Skipping libnuma benchmarks: need GPUs on >=2 NUMA nodes");
    }
    run_cuda_pinned_benchmarks(&streams);
}
