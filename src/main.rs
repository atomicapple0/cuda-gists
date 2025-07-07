use cudarc::driver::DriverError;
use cudarc::driver::sys;
use std::mem::MaybeUninit;

mod log;

fn setup() -> Result<*mut sys::CUctx_st, DriverError> {
    unsafe { sys::cuInit(0) }.result().unwrap();

    let dev = unsafe {
        let mut pdev = MaybeUninit::uninit();
        sys::cuDeviceGet(pdev.as_mut_ptr(), 0).result().unwrap();
        pdev.assume_init()
    };
    log!("dev: {:?}", dev);

    let ctx = unsafe {
        let mut pctx = MaybeUninit::uninit();
        sys::cuCtxCreate_v2(pctx.as_mut_ptr(), 0, dev)
            .result()
            .unwrap();
        pctx.assume_init()
    };
    log!("ctx created: {:?}", ctx);

    Ok(ctx)
}

fn bytes_to_human_readable(bytes_usize: usize) -> String {
    let mut running = bytes_usize as f64;
    if running < 1024.0 {
        return format!("{:.2} B", running);
    }
    running /= 1024.0;
    if running < 1024.0 {
        return format!("{:.2} KB", running);
    }
    running /= 1024.0;
    if running < 1024.0 {
        return format!("{:.2} MB", running);
    }
    running /= 1024.0;
    if running < 1024.0 {
        return format!("{:.2} GB", running);
    }
    running /= 1024.0;
    if running < 1024.0 {
        return format!("{:.2} TB", running);
    }
    format!("{:.2} PB", running / 1024.0)
}

fn fn1() -> Result<(), DriverError> {
    log!("Hello from fn1");

    let _ = setup()?;

    let prop = sys::CUmemAllocationProp {
        type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes: sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_FABRIC,
        location: sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id: 0,
        },
        allocFlags: sys::CUmemAllocationProp_st__bindgen_ty_1 {
            compressionType: 0,
            gpuDirectRDMACapable: 1,
            usage: 0,
            reserved: [0; 4],
        },
        win32HandleMetaData: std::ptr::null_mut(),
    };

    let granularity = unsafe {
        let mut pgranularity = MaybeUninit::uninit();
        let opt = sys::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM;
        sys::cuMemGetAllocationGranularity(pgranularity.as_mut_ptr(), &prop, opt)
            .result()
            .unwrap();
        pgranularity.assume_init()
    };
    log!("granularity: 0x{:x?}", granularity);
    log!("granularity: {}", bytes_to_human_readable(granularity));

    // cuMemCreate
    let size = 0x200000;

    log!("prop: {:#?}", prop);

    let mem = unsafe {
        let mut phandle = MaybeUninit::uninit();
        sys::cuMemCreate(phandle.as_mut_ptr(), size, &prop, 0)
            .result()
            .unwrap();
        phandle.assume_init()
    };
    log!("mem created: {:x?}", mem);

    Ok(())
}

fn fn2() {
    // set env var CUDA_VISIBLE_DEVICES=0,1,2,3
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", "0,1") };

    unsafe { sys::cuInit(0) }.result().unwrap();

    let n_devices = unsafe {
        let mut count = 0;
        sys::cuDeviceGetCount(&mut count).result().unwrap();
        count as usize
    };
    log!("n_devices: {}", n_devices);

    let ctxs = unsafe {
        let mut ctxs: [*mut sys::CUctx_st; 10] = [std::ptr::null_mut(); 10];
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

fn main() {
    fn2();
}
