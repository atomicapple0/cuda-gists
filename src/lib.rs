use cudarc::driver::sys;
use std::mem::MaybeUninit;
use std::sync::LazyLock;

pub mod log;

pub fn cu_init() {
    unsafe { sys::cuInit(0) }.result().unwrap();
}

pub static INIT: LazyLock<()> = LazyLock::new(|| {
    log!("Initializing CUDA");
    cu_init();
});

#[derive(Debug)]
pub struct Context {
    pub id: i32,
    pub ctx: *mut sys::CUctx_st,
}

impl Context {
    pub fn new(id: i32) -> Self {
        _ = *INIT;

        let dev = unsafe {
            let mut pdev = MaybeUninit::uninit();
            sys::cuDeviceGet(pdev.as_mut_ptr(), 0).result().unwrap();
            pdev.assume_init()
        };

        let ctx = unsafe {
            let mut pctx = MaybeUninit::uninit();
            sys::cuCtxCreate_v2(pctx.as_mut_ptr(), 0, dev)
                .result()
                .unwrap();
            pctx.assume_init()
        };

        let ctx = Self { id, ctx };
        log!("Created {:?}", ctx);
        ctx
    }
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
