// EVOLVE-BLOCK-START
pub fn core_add(a: i32, b: i32) -> i32 {
    a + b
}
// EVOLVE-BLOCK-END

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_core_add() { assert_eq!(core_add(1, 2), 3); }
}
