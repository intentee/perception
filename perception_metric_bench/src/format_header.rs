#[must_use]
pub fn format_header() -> String {
    format!(
        "{:<8} {:>11} {:>12} {:>12} {:>12}",
        "kind", "size", "median ms", "min ms", "Mpx/s"
    )
}

#[cfg(test)]
mod tests {
    use super::format_header;

    #[test]
    fn lays_out_the_five_column_titles() {
        let header = format_header();

        assert!(header.starts_with("kind"));
        assert!(header.contains("median ms"));
        assert!(header.contains("Mpx/s"));
    }
}
