use cudarc::driver::DriverError;
use cudarc::driver::sys;
use std::mem::MaybeUninit;

pub mod log;

pub fn setup() -> Result<*mut sys::CUctx_st, DriverError> {
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

pub fn bytes_to_human_readable(bytes_usize: usize) -> String {
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
