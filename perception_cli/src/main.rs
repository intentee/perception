use anyhow::Result;

use perception_cli::run::run;

fn main() -> Result<()> {
    run(std::env::args_os().collect())
}

#[cfg(test)]
mod tests {
    #[test]
    fn reports_an_error_for_the_test_harness_arguments() {
        super::main()
            .expect_err("the test harness arguments are not a valid perception invocation");
    }
}
