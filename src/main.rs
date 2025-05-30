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

fn fn1() -> Result<(), DriverError> {
    log!("Hello from fn1");

    let _ = setup()?;

    // cuMemCreate
    let size = 1024;
    let prop = sys::CUmemAllocationProp {
        type_: sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        requestedHandleTypes:
            sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
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
