//! Shared GPU format codes, row-byte geometry, and launch helpers used by both
//! the Metal and CUDA backends.

use allpaka_dtype::GgmlType;

/// Unified kernel ABI format code for GPU row dequantization (`dequant_rows_f16`,
/// `embed_row_f32`, and `embed_row_f32_dtoken`).
///
/// Both Metal (`shaders/kernels.metal`) and CUDA (`cuda/kernels.rs`) switch on
/// these exact integer values (`0..=9`).
#[inline]
pub const fn dq_fmt(ty: GgmlType) -> Option<u32> {
    Some(match ty {
        GgmlType::Q4K => 0,
        GgmlType::Q5K => 1,
        GgmlType::Q6K => 2,
        GgmlType::Q8_0 => 3,
        GgmlType::Q2K => 4,
        GgmlType::Q3K => 5,
        GgmlType::Q5_0 => 6,
        GgmlType::F16 => 7,
        GgmlType::F32 => 8,
        GgmlType::BF16 => 9,
        GgmlType::MlxQ4 | GgmlType::MlxQ8 | GgmlType::Other(_) => return None,
    })
}

/// Exact byte stride of one quantized/dense row of `n_in` elements, or `None`
/// if `ty` is unsupported or `n_in` is not a multiple of its block width.
#[inline]
pub fn row_bytes(ty: GgmlType, n_in: usize) -> Option<usize> {
    ty.expected_bytes(n_in).ok()
}

/// Number of 32-lane threadgroups/warps to launch for `embed_row_f32*` given
/// unified `fmt` (`0..=9`) and `n_in` elements.
///
/// Formats `0..=5` (`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q2_K`, `Q3_K`) assign up to
/// 256 elements per 32-lane group; formats `6..=9` (`Q5_0`, `F16`, `F32`,
/// `BF16`) assign 32 elements per 32-lane group.
#[inline]
pub const fn embed_row_groups(fmt: u32, n_in: usize) -> u32 {
    let g = if fmt <= 5 {
        n_in.div_ceil(256)
    } else {
        n_in.div_ceil(32)
    };
    if g == 0 {
        1
    } else {
        g as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dq_fmt_covers_all_ten_gguf_types_uniquely() {
        let cases = [
            (GgmlType::Q4K, 0),
            (GgmlType::Q5K, 1),
            (GgmlType::Q6K, 2),
            (GgmlType::Q8_0, 3),
            (GgmlType::Q2K, 4),
            (GgmlType::Q3K, 5),
            (GgmlType::Q5_0, 6),
            (GgmlType::F16, 7),
            (GgmlType::F32, 8),
            (GgmlType::BF16, 9),
        ];
        for (ty, code) in cases {
            assert_eq!(dq_fmt(ty), Some(code), "{ty:?}");
            assert!(row_bytes(ty, 256).is_some(), "{ty:?} row_bytes(256)");
        }
        assert_eq!(dq_fmt(GgmlType::MlxQ4), None);
        assert_eq!(dq_fmt(GgmlType::MlxQ8), None);
        assert_eq!(dq_fmt(GgmlType::Other(77)), None);
        assert_eq!(row_bytes(GgmlType::Q4K, 128), None);
        assert_eq!(row_bytes(GgmlType::Q5_0, 128), Some(88));
    }

    #[test]
    fn embed_row_groups_matches_format_granularity() {
        for fmt in 0..=5 {
            assert_eq!(embed_row_groups(fmt, 512), 2);
            assert_eq!(embed_row_groups(fmt, 128), 1);
        }
        for fmt in 6..=9 {
            assert_eq!(embed_row_groups(fmt, 512), 16);
            assert_eq!(embed_row_groups(fmt, 128), 4);
        }
    }
}
