use anyhow::{Context, Result};
use std::fs::File;
use std::io;
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::os::fd::AsRawFd;
use std::path::Path;

/// Load little-endian f32 words without first zeroing the full destination.
pub(super) fn load(path: &Path, states: u64) -> Result<Vec<f32>> {
    let expected_bytes = states
        .checked_mul(4)
        .context("f32 RAM-load expected size overflow")?;
    let len = usize::try_from(states).context("state count too large for usize")?;
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let actual = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    if actual != expected_bytes {
        anyhow::bail!(
            "f32 RAM-load size mismatch {}: got {}, expected {}",
            path.display(),
            actual,
            expected_bytes
        );
    }

    // SAFETY: libc::read writes the returned number of bytes into the supplied
    // allocation, never reads them, and does not retain the pointer. The file
    // stays open throughout the read loop.
    unsafe {
        read_values_with(len, |destination, count| {
            let result = libc::read(file.as_raw_fd(), destination.cast(), count);
            if result < 0 {
                Err(io::Error::last_os_error())
            } else {
                Ok(result as usize)
            }
        })
    }
    .with_context(|| format!("failed to read {} into RAM", path.display()))
}

/// Read into a raw allocation and expose f32 values only once fully initialized.
///
/// # Safety
/// On each call, `read` must write the first `n` destination bytes before
/// returning `Ok(n)`, with `n <= count`. It may not read the destination bytes,
/// write outside the requested range, or retain the pointer after returning.
unsafe fn read_values_with(
    len: usize,
    mut read: impl FnMut(*mut u8, usize) -> io::Result<usize>,
) -> io::Result<Vec<f32>> {
    let byte_len = len
        .checked_mul(mem::size_of::<f32>())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "f32 byte length overflow"))?;
    let mut storage = Vec::<MaybeUninit<f32>>::new();
    storage.try_reserve_exact(len).map_err(io::Error::other)?;
    let destination = storage.as_mut_ptr().cast::<u8>();
    let mut initialized = 0;
    while initialized < byte_len {
        // POSIX read counts must fit ssize_t. Kernels may return a smaller
        // successful read, so even large regular files use the same loop.
        let count = (byte_len - initialized).min(isize::MAX as usize);
        // SAFETY: the offset stays inside the reserved allocation until the
        // loop completes, and the callback follows the contract above.
        let next = unsafe { destination.add(initialized) };
        match read(next, count) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole f32 buffer",
                ));
            }
            Ok(n) if n <= count => initialized += n,
            Ok(_) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "read returned more bytes than requested",
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }

    // Every byte is initialized now; every f32 bit pattern is valid, including
    // NaNs. MaybeUninit<f32> has exactly the layout and alignment of f32, so
    // ownership and the original allocation capacity transfer unchanged.
    let mut storage = ManuallyDrop::new(storage);
    let values =
        unsafe { Vec::from_raw_parts(storage.as_mut_ptr().cast::<f32>(), len, storage.capacity()) };
    #[cfg(target_endian = "big")]
    let values = {
        let mut values = values;
        for value in &mut values {
            *value = f32::from_bits(u32::from_le(value.to_bits()));
        }
        values
    };
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct TempFile(std::path::PathBuf);

    impl TempFile {
        fn new(bytes: &[u8]) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "zxcl-f32-ram-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&path, bytes).unwrap();
            Self(path)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn loads_empty_file_without_reading() {
        let file = TempFile::new(&[]);
        assert!(load(&file.0, 0).unwrap().is_empty());
        // SAFETY: the callback is never called for an empty destination.
        let values = unsafe { read_values_with(0, |_, _| panic!("unexpected empty read")) };
        assert!(values.unwrap().is_empty());
    }

    #[test]
    fn loads_exact_little_endian_bits_including_nonfinite_values() {
        let bits = [
            0x0000_0000u32,
            0x8000_0000,
            0x3f80_0000,
            0xbf00_0000,
            0x0000_0001,
            0x7f80_0000,
            0xff80_0000,
            0x7fc1_2345,
            0xff81_2345,
        ];
        let bytes = bits
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let file = TempFile::new(&bytes);
        let values = load(&file.0, bits.len() as u64).unwrap();
        assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    }

    #[test]
    fn rejects_short_and_long_files_before_loading() {
        for bytes in [&[0u8; 3][..], &[0u8; 5][..]] {
            let file = TempFile::new(bytes);
            let error = load(&file.0, 1).unwrap_err();
            assert!(error.to_string().contains("size mismatch"));
        }
        let file = TempFile::new(&[]);
        assert!(load(&file.0, u64::MAX)
            .unwrap_err()
            .to_string()
            .contains("expected size overflow"));
    }

    #[test]
    fn retries_interrupted_and_partial_reads_across_word_boundaries() {
        let bits = [0x3f80_0000u32, 0x8000_0000, 0x7fa1_2345];
        let bytes = bits
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        let mut offset = 0;
        let mut calls = 0;
        // SAFETY: each successful callback copies exactly its return count of
        // initialized bytes, staying within the requested destination range.
        let values = unsafe {
            read_values_with(bits.len(), |destination, count| {
                calls += 1;
                if calls == 1 || calls == 4 {
                    return Err(io::Error::from(io::ErrorKind::Interrupted));
                }
                let n = count.min(3).min(bytes.len() - offset);
                std::ptr::copy_nonoverlapping(bytes.as_ptr().add(offset), destination, n);
                offset += n;
                Ok(n)
            })
        }
        .unwrap();
        assert_eq!(offset, bytes.len());
        assert!(calls > 4);
        assert_eq!(values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), bits);
    }

    #[test]
    fn partial_initialization_is_discarded_on_eof_or_read_error() {
        for kind in [
            io::ErrorKind::UnexpectedEof,
            io::ErrorKind::PermissionDenied,
        ] {
            let mut calls = 0;
            // SAFETY: the only successful nonempty read writes three bytes.
            // Subsequent calls initialize nothing and return EOF or an error.
            let error = unsafe {
                read_values_with(2, |destination, count| {
                    calls += 1;
                    if calls == 1 {
                        assert_eq!(count, 8);
                        std::ptr::copy_nonoverlapping([1u8, 2, 3].as_ptr(), destination, 3);
                        return Ok(3);
                    }
                    assert_eq!(count, 5);
                    if kind == io::ErrorKind::UnexpectedEof {
                        Ok(0)
                    } else {
                        Err(io::Error::from(kind))
                    }
                })
            }
            .unwrap_err();
            assert_eq!(error.kind(), kind);
            assert_eq!(calls, 2);
        }
    }
}
