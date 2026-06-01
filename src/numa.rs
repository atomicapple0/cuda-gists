//! NUMA node discovery and libnuma-backed host buffers for GPU<->host benchmarks.

use std::ffi::c_void;
use std::io;
use std::process::Command;

use cudarc::driver::sys;
use libc;

use crate::{AddressSpace, Buffer, Context, INIT};

#[link(name = "numa")]
unsafe extern "C" {
    fn numa_available() -> i32;
    fn numa_alloc_onnode(size: usize, node: i32) -> *mut c_void;
    fn numa_free(ptr: *mut c_void, size: usize);
}

/// GPU index and the CPU NUMA node its PCI device is attached to.
#[derive(Debug, Clone, Copy)]
pub struct GpuNumaInfo {
    pub device_id: i32,
    pub numa_node: i32,
}

/// Host memory allocated on a specific NUMA node and registered with CUDA for DMA.
pub struct NumaHostBuffer {
    ptr: *mut c_void,
    size: usize,
    node: i32,
}

impl NumaHostBuffer {
    /// Allocate `size` bytes on `node`, fault pages there, and pin via `cuMemHostRegister`.
    pub fn alloc_onnode(size: usize, node: i32) -> Self {
        _ = *INIT;
        if unsafe { numa_available() } < 0 {
            panic!("libnuma reports NUMA is not available on this host");
        }
        let ptr = unsafe { numa_alloc_onnode(size, node) };
        if ptr.is_null() {
            panic!("numa_alloc_onnode({size}, node={node}) failed");
        }
        // Touch every page so physical memory is bound to `node` before timing.
        unsafe { libc::memset(ptr, 0, size) };
        unsafe {
            sys::cuMemHostRegister_v2(ptr, size, 0)
                .result()
                .unwrap_or_else(|e| {
                    numa_free_unregistered(ptr, size);
                    panic!("cuMemHostRegister_v2(node={node}, size={size}) failed: {e:?}");
                });
        }
        Self { ptr, size, node }
    }

    pub fn node(&self) -> i32 {
        self.node
    }

    /// View as a [`Buffer`] for use with [`Stream::memcpy_async`].
    pub fn as_buffer(&self, ctx: &Context) -> Buffer {
        Buffer {
            ctx: ctx.clone(),
            size: self.size,
            address_space: AddressSpace::Pinned,
            addr: self.ptr as u64,
        }
    }

    pub fn free(self) {
        if !self.ptr.is_null() {
            unsafe {
                sys::cuMemHostUnregister(self.ptr)
                    .result()
                    .expect("cuMemHostUnregister failed");
                numa_free(self.ptr, self.size);
            }
        }
    }
}

fn numa_free_unregistered(ptr: *mut c_void, size: usize) {
    unsafe { numa_free(ptr, size) };
}

/// Map an nvidia-smi PCI bus id to its CPU NUMA node via sysfs.
fn sysfs_numa_node(pci_bus_id: &str) -> i32 {
    let pci_lower = pci_bus_id.trim().to_lowercase();
    let parts: Vec<&str> = pci_lower.split(':').collect();
    if parts.len() != 3 {
        return -1;
    }
    let domain_raw = parts[0];
    let domain = if domain_raw.len() >= 4 {
        &domain_raw[domain_raw.len() - 4..]
    } else {
        domain_raw
    };
    let domain = format!("{domain:0>4}");
    let path = format!(
        "/sys/bus/pci/devices/{}:{}:{}/numa_node",
        domain, parts[1], parts[2]
    );
    match std::fs::read_to_string(&path) {
        Ok(s) => s.trim().parse().unwrap_or(-1),
        Err(_) => -1,
    }
}

/// Discover GPU index -> NUMA node mapping via nvidia-smi + sysfs.
pub fn discover_gpu_numa_nodes() -> io::Result<Vec<GpuNumaInfo>> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
        .output()?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    let mut gpus = Vec::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (idx_str, pci) = line
            .split_once(',')
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, line.to_string()))?;
        gpus.push(GpuNumaInfo {
            device_id: idx_str.trim().parse().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("bad gpu index: {idx_str}"),
                )
            })?,
            numa_node: sysfs_numa_node(pci.trim()),
        });
    }
    Ok(gpus)
}

/// Return `(node_a, node_b)` for the first two NUMA nodes that have GPUs attached.
pub fn two_gpu_numa_domains(gpus: &[GpuNumaInfo]) -> Option<(i32, i32)> {
    let mut nodes: Vec<i32> = gpus
        .iter()
        .map(|g| g.numa_node)
        .filter(|&n| n >= 0)
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.len() < 2 {
        return None;
    }
    Some((nodes[0], nodes[1]))
}

pub fn other_node(gpu_node: i32, node_a: i32, node_b: i32) -> i32 {
    if gpu_node == node_a { node_b } else { node_a }
}
