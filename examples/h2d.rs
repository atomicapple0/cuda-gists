use itertools::izip;

use cuda_gists::*;

const ITERS: usize = 3;
const GB: usize = 1024 * 1024 * 1024;

pub fn compute_bandwidth_gb_s(time: std::time::Duration, size: usize) -> f64 {
    let gb = size as f64 / 1024.0 / 1024.0 / 1024.0;
    gb / time.as_secs_f64()
}

fn main() {
    log!("Hello from h2d");

    const NUM_DEVICES: usize = 4;
    let ctxs = (0..NUM_DEVICES)
        .map(|i| Context::new(i as i32))
        .collect::<Vec<_>>();

    let streams = ctxs
        .iter()
        .map(|ctx| ctx.create_stream())
        .collect::<Vec<_>>();

    let pageable_bufs = streams[0].create_buffer_async(8 * GB, AddressSpace::Cpu);

    let pinned_bufs = streams
        .iter()
        .map(|stream| stream.create_buffer_async(8 * GB, AddressSpace::Pinned))
        .collect::<Vec<_>>();

    let gpu_bufs = streams
        .iter()
        .map(|stream| stream.create_buffer_async(8 * GB, AddressSpace::Device))
        .collect::<Vec<_>>();

    let events = ctxs
        .iter()
        .map(|ctx| ctx.create_event())
        .collect::<Vec<_>>();

    for stream in &streams {
        stream.synchronize();
    }

    log!("Benchmarking GPU0 -> GPU1 P2P");
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        streams[0].memcpy_async(&gpu_bufs[1], &gpu_bufs[0]);
        let t1 = std::time::Instant::now();
        for stream in &streams {
            stream.synchronize();
        }
        let t2 = std::time::Instant::now();

        let copy_time = t1.duration_since(t0);
        let sync_time = t2.duration_since(t1);
        let total_time = t2.duration_since(t0);
        let bw = compute_bandwidth_gb_s(total_time, 8 * GB);
        log!(
            "--- Copy time: {:?}, Sync time: {:?}, Bandwidth: {:.2} GB/s",
            copy_time,
            sync_time,
            bw
        );
    }

    log!("Benchmarking Pageable -> GPU0");
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        streams[0].memcpy_async(&gpu_bufs[0], &pageable_bufs);
        let t1 = std::time::Instant::now();
        for stream in &streams {
            stream.synchronize();
        }
        let t2 = std::time::Instant::now();

        let copy_time = t1.duration_since(t0);
        let sync_time = t2.duration_since(t1);
        let total_time = t2.duration_since(t0);
        let bw = compute_bandwidth_gb_s(total_time, 8 * GB);
        log!(
            "--- Copy time: {:?}, Sync time: {:?}, Bandwidth: {:.2} GB/s",
            copy_time,
            sync_time,
            bw
        );
    }

    log!("Benchmarking Pinned0 -> GPU0");
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        streams[0].memcpy_async(&gpu_bufs[0], &pinned_bufs[0]);
        let t1 = std::time::Instant::now();
        for stream in &streams {
            stream.synchronize();
        }
        let t2 = std::time::Instant::now();

        let copy_time = t1.duration_since(t0);
        let sync_time = t2.duration_since(t1);
        let total_time = t2.duration_since(t0);
        let bw = compute_bandwidth_gb_s(total_time, 8 * GB);
        log!(
            "--- Copy time: {:?}, Sync time: {:?}, Bandwidth: {:.2} GB/s",
            copy_time,
            sync_time,
            bw
        );
    }

    log!("Benchmarking Pinned0 -> GPU1");
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        streams[1].memcpy_async(&gpu_bufs[1], &pinned_bufs[0]);
        let t1 = std::time::Instant::now();
        for stream in &streams {
            stream.synchronize();
        }
        let t2 = std::time::Instant::now();

        let copy_time = t1.duration_since(t0);
        let sync_time = t2.duration_since(t1);
        let total_time = t2.duration_since(t0);
        let bw = compute_bandwidth_gb_s(total_time, 8 * GB);
        log!(
            "--- Copy time: {:?}, Sync time: {:?}, Bandwidth: {:.2} GB/s",
            copy_time,
            sync_time,
            bw
        );
    }

    log!("Benchmarking Pageable -> Pinned0");
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        streams[0].memcpy_async(&pinned_bufs[0], &pageable_bufs);
        let t1 = std::time::Instant::now();
        for stream in &streams {
            stream.synchronize();
        }
        let t2 = std::time::Instant::now();

        let copy_time = t1.duration_since(t0);
        let sync_time = t2.duration_since(t1);
        let total_time = t2.duration_since(t0);
        let bw = compute_bandwidth_gb_s(total_time, 8 * GB);
        log!(
            "--- Copy time: {:?}, Sync time: {:?}, Bandwidth: {:.2} GB/s",
            copy_time,
            sync_time,
            bw
        );
    }

    log!("Benchmarking Pinned0 -> GPU0, Pinned1 -> GPU1, Pinned2 -> GPU2, ... (multi stream)");
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        for (pinned_buf, gpu_buf, stream) in izip!(&pinned_bufs, &gpu_bufs, &streams) {
            stream.memcpy_async(gpu_buf, pinned_buf);
        }
        let t1 = std::time::Instant::now();
        for stream in &streams {
            stream.synchronize();
        }
        let t2 = std::time::Instant::now();

        let copy_time = t1.duration_since(t0);
        let sync_time = t2.duration_since(t1);
        log!("--- Copy time: {:?}, Sync time: {:?}", copy_time, sync_time);
    }

    log!("Benchmarking Pinned0 -> GPU0, GPU1, GPU2, ... (multi stream)");
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        let pinned0 = pinned_bufs[0].clone();
        for (stream, gpu_buf) in izip!(&streams, &gpu_bufs) {
            stream.memcpy_async(gpu_buf, &pinned0);
        }
        let t1 = std::time::Instant::now();
        for stream in &streams {
            stream.synchronize();
        }
        let t2 = std::time::Instant::now();

        let copy_time = t1.duration_since(t0);
        let sync_time = t2.duration_since(t1);
        log!("--- Copy time: {:?}, Sync time: {:?}", copy_time, sync_time);
    }

    log!("Benchmarking Pinned -> GPU0 -> GPU1, GPU2, GPU3, ... (multi stream)");
    for _ in 0..ITERS {
        let t0 = std::time::Instant::now();
        streams[0].memcpy_async(&gpu_bufs[0], &pinned_bufs[0]);
        streams[0].record_event(&events[0]);
        for (stream, event, gpu_buf) in izip!(&streams, &events, &gpu_bufs).skip(1) {
            stream.wait_for_event(event);
            stream.memcpy_async(gpu_buf, &gpu_bufs[0]);
        }

        let t1 = std::time::Instant::now();
        for stream in &streams {
            stream.synchronize();
        }
        let t2 = std::time::Instant::now();

        let copy_time = t1.duration_since(t0);
        let sync_time = t2.duration_since(t1);
        log!("--- Copy time: {:?}, Sync time: {:?}", copy_time, sync_time);
    }
}
