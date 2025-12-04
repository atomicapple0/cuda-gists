use cudarc::driver::DriverError;
use cudarc::driver::sys;
use std::mem::MaybeUninit;

use cuda_gists::*;

fn main() -> Result<(), DriverError> {
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
