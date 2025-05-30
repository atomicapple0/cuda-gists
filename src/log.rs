#[macro_export]
macro_rules! log {
    ($($x:tt)*) => {
        {
            use colored::Colorize;
            let s = format!("[{}|{}:{}] ", std::process::id(), file!(), line!());
            eprint!("{}", s.cyan());
            eprintln!($($x)*)
        }
    }
}
