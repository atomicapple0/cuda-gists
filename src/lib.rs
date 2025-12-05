use cudarc::driver::sys;
use libc;
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

#[derive(Debug, Clone, PartialEq)]
pub enum AddressSpace {
    Device,
    Pinned,
    Cpu,
}

#[derive(Debug, Clone)]
pub struct Buffer {
    pub ctx: Context,
    pub size: usize,
    pub address_space: AddressSpace,
    pub addr: u64,
}

#[derive(Debug, Clone)]
pub struct Stream {
    pub ctx: Context,
    pub stream: *mut sys::CUstream_st,
}

impl Stream {
    pub fn create_buffer_async(&self, size: usize, address_space: AddressSpace) -> Buffer {
        self.ctx.set_current();
        let addr = match address_space {
            AddressSpace::Device => unsafe {
                let mut pbuffer = MaybeUninit::uninit();
                sys::cuMemAlloc_v2(pbuffer.as_mut_ptr(), size)
                    .result()
                    .unwrap();
                pbuffer.assume_init()
            },
            AddressSpace::Pinned => unsafe {
                let mut pbuffer = MaybeUninit::uninit();
                sys::cuMemAllocHost_v2(pbuffer.as_mut_ptr(), size)
                    .result()
                    .unwrap();
                pbuffer.assume_init() as u64
            },
            AddressSpace::Cpu => unsafe { libc::malloc(size) as u64 },
        };
        Buffer {
            ctx: self.ctx.clone(),
            size,
            address_space,
            addr,
        }
    }

    pub fn free_buffer_sync(&self, buf: &Buffer) {
        self.ctx.set_current();
        match buf.address_space {
            AddressSpace::Device => unsafe { sys::cuMemFree_v2(buf.addr) }.result().unwrap(),
            AddressSpace::Pinned => unsafe { sys::cuMemFreeHost(buf.addr as *mut libc::c_void) }
                .result()
                .unwrap(),
            AddressSpace::Cpu => unsafe { libc::free(buf.addr as *mut libc::c_void) },
        }
    }

    pub fn memcpy_async(&self, dst: &Buffer, src: &Buffer) {
        self.ctx.set_current();
        if dst.size != src.size {
            panic!("size mismatch");
        }

        // if dst.address_space == AddressSpace::Device
        //     && src.address_space == AddressSpace::Device
        //     && dst.ctx.device_id != src.ctx.device_id
        // {
        //     unsafe {
        //         sys::cuMemcpyPeerAsync(
        //             dst.addr,
        //             dst.ctx.ctx,
        //             src.addr,
        //             src.ctx.ctx,
        //             src.size,
        //             self.stream,
        //         )
        //         .result()
        //         .unwrap();
        //     }
        //     return;
        // }

        // log!("Copying from {:?} to {:?} on {:?}", src, dst, self);
        unsafe { sys::cuMemcpyAsync(dst.addr, src.addr, src.size, self.stream) }
            .result()
            .unwrap();
    }

    pub fn synchronize(&self) {
        self.ctx.set_current();
        // log!("Synchronizing stream {:?}", self);
        unsafe { sys::cuStreamSynchronize(self.stream) }
            .result()
            .unwrap();
    }

    pub fn record_event(&self, event: &Event) {
        self.ctx.set_current();
        unsafe { sys::cuEventRecord(event.event, self.stream) }
            .result()
            .unwrap();
    }

    pub fn wait_for_event(&self, event: &Event) {
        self.ctx.set_current();
        unsafe { sys::cuStreamWaitEvent(self.stream, event.event, 0) }
            .result()
            .unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct Context {
    pub ctx: *mut sys::CUctx_st,
    pub device_id: i32,
}

#[derive(Debug, Clone)]
pub struct Event {
    pub ctx: Context,
    pub event: *mut sys::CUevent_st,
}

impl Context {
    pub fn new(device_id: i32) -> Self {
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

        let ctx = Self { ctx, device_id };
        log!("Created {:?}", ctx);
        ctx
    }

    pub fn set_current(&self) {
        unsafe { sys::cuCtxSetCurrent(self.ctx) }.result().unwrap();
    }

    pub fn create_stream(&self) -> Stream {
        self.set_current();
        let stream = unsafe {
            let mut pstream = MaybeUninit::uninit();
            sys::cuStreamCreate(
                pstream.as_mut_ptr(),
                sys::CUstream_flags_enum::CU_STREAM_DEFAULT as u32,
            )
            .result()
            .unwrap();
            pstream.assume_init()
        };
        log!("Created {:?} on {:?}", stream, self);
        Stream {
            ctx: self.clone(),
            stream,
        }
    }

    pub fn create_event(&self) -> Event {
        self.set_current();
        let event = unsafe {
            let mut pevent = MaybeUninit::uninit();
            sys::cuEventCreate(
                pevent.as_mut_ptr(),
                sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32,
            )
            .result()
            .unwrap();
            pevent.assume_init()
        };
        Event {
            ctx: self.clone(),
            event,
        }
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
