use cudarc::driver::DriverError;
use cudarc::driver::sys::*;

mod log;

fn fn1() -> Result<(), DriverError> {
    log!("Hello from fn1");
    unsafe { cuInit(0).result() }?;
    Ok(())
}

fn main() -> Result<(), DriverError> {
    fn1()?;
    Ok(())
}
