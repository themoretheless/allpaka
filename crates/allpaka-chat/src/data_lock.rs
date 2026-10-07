//! A process-scoped OS lock prevents concurrent Studio writers and false recovery.
use anyhow::{bail, Context, Result};
use fs2::FileExt;
use std::{
    fs::{File, OpenOptions},
    path::Path,
};

pub(crate) struct DataLock {
    _file: File,
}
impl Drop for DataLock {
    fn drop(&mut self) {
        // A concurrently spawned child may briefly inherit this open file
        // before exec closes it. Explicit unlock avoids waiting for that copy.
        if let Err(error) = FileExt::unlock(&self._file) {
            eprintln!("Studio directory unlock failed: {error}");
        }
    }
}
impl DataLock {
    pub(crate) fn acquire(data: &Path) -> Result<Self> {
        let path = data.join(".studio.lock");
        if std::fs::symlink_metadata(&path).is_ok_and(|m| !m.is_file()) {
            bail!("Invalid Studio directory lock");
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        file.try_lock_exclusive().context("Another Studio process owns this data directory; use a different --data-dir or stop that process")?;
        Ok(Self { _file: file })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn unlocking_is_not_delayed_by_a_forked_child() {
        let data = std::env::temp_dir().join(crate::evaluation::new_id());
        std::fs::create_dir(&data).unwrap();
        let first = DataLock::acquire(&data).unwrap();
        let mut pipe = [0; 2];
        assert_eq!(unsafe { libc::pipe(pipe.as_mut_ptr()) }, 0);
        let child = unsafe { libc::fork() };
        assert!(child >= 0);
        if child == 0 {
            // After fork in a multithreaded test process, only async-signal-safe calls.
            unsafe {
                libc::close(pipe[1]);
                let mut signal = 0u8;
                libc::read(pipe[0], (&mut signal as *mut u8).cast(), 1);
                libc::_exit(0);
            }
        }
        unsafe {
            libc::close(pipe[0]);
        }
        drop(first);
        let acquired = DataLock::acquire(&data);
        unsafe {
            let signal = 1u8;
            libc::write(pipe[1], (&signal as *const u8).cast(), 1);
            libc::close(pipe[1]);
            libc::waitpid(child, std::ptr::null_mut(), 0);
        }
        assert!(
            acquired.is_ok(),
            "child's inherited descriptor retained the parent's lock"
        );
        drop(acquired);
        std::fs::remove_dir_all(data).unwrap();
    }
    #[test]
    fn lock_is_exclusive_and_released_without_removing_the_file() {
        let data = std::env::temp_dir().join(crate::evaluation::new_id());
        std::fs::create_dir(&data).unwrap();
        let first = DataLock::acquire(&data).unwrap();
        assert!(DataLock::acquire(&data).is_err());
        drop(first);
        let second = DataLock::acquire(&data).unwrap();
        drop(second);
        assert!(data.join(".studio.lock").exists());
        std::fs::remove_dir_all(data).unwrap();
    }
}
