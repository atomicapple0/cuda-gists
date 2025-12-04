use cudarc::driver::sys;

use cuda_gists::*;

fn main() {
    // set env var CUDA_VISIBLE_DEVICES=0,1,2,3
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "0,1") };

    unsafe { sys::cuInit(0) }.result().unwrap();

    let n_devices = unsafe {
        let mut count = 0;
        sys::cuDeviceGetCount(&mut count).result().unwrap();
        count as usize
    };
    log!("n_devices: {}", n_devices);

    const MAX_NUM_DEVICES: usize = 10;
    assert!(n_devices <= MAX_NUM_DEVICES);
    let ctxs = unsafe {
        let mut ctxs: [*mut sys::CUctx_st; MAX_NUM_DEVICES] =
            [std::ptr::null_mut(); MAX_NUM_DEVICES];
        for i in 0..n_devices {
            sys::cuCtxCreate_v2(&mut ctxs[i], 0, i as i32)
                .result()
                .unwrap();
        }
        ctxs
    };
    log!("ctxs: {:?}", ctxs);

    for i in 0..n_devices {
        for j in 0..n_devices {
            if i == j {
                continue;
            }
            let ctx = ctxs[i];
            let dev = ctxs[j];
            log!("ctx: {:?}, dev: {:?}", ctx, dev);
            // check if can access peer
            let can_access = unsafe {
                let mut can_access = 0;
                sys::cuDeviceCanAccessPeer(&mut can_access, i as i32, j as i32)
                    .result()
                    .unwrap();
                can_access
            };
            log!(
                "cuDeviceCanAccessPeer {i} -> {j}: {}",
                if can_access == 1 { "yes" } else { "no" }
            );
            if can_access == 1 {
                unsafe {
                    sys::cuCtxSetCurrent(ctxs[i]).result().unwrap();
                    sys::cuCtxEnablePeerAccess(ctxs[j], 0).result().unwrap();
                }
                log!("enabled peer access {i} -> {j}");
            }
        }
    }
}
