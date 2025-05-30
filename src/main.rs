use cudarc::driver::DriverError;
use cudarc::driver::sys;
use std::mem::MaybeUninit;

mod log;

fn setup() -> Result<*mut sys::CUctx_st, DriverError> {
    unsafe { sys::cuInit(0) }.result()?;

    let dev = unsafe {
        let mut pdev = MaybeUninit::uninit();
        sys::cuDeviceGet(pdev.as_mut_ptr(), 0).result()?;
        pdev.assume_init()
    };
    log!("dev: {:?}", dev);

    let ctx = unsafe {
        let mut pctx = MaybeUninit::uninit();
        sys::cuCtxCreate_v2(pctx.as_mut_ptr(), 0, dev).result()?;
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
        sys::cuMemGetAllocationGranularity(pgranularity.as_mut_ptr(), &prop, opt).result()?;
        pgranularity.assume_init()
    };
    log!("granularity: 0x{:x?}", granularity);
    log!("granularity: {}", bytes_to_human_readable(granularity));

    // cuMemCreate
    let size = 0x200000;

    log!("prop: {:?}", prop);

    let mem = unsafe {
        let mut phandle = MaybeUninit::uninit();
        sys::cuMemCreate(phandle.as_mut_ptr(), size, &prop, 0).result()?;
        phandle.assume_init()
    };
    log!("mem created: {:?}", mem);

    Ok(())
}

fn main() -> Result<(), DriverError> {
    fn1()?;
    Ok(())
}
