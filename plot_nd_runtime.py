from gitbud.gitbud import inject_repo_into_sys_path

inject_repo_into_sys_path()


def main() -> None:
    raise RuntimeError(
        "Deprecated script. Use the Makefile targets and new v2 modules under benchmarks/ and plots/."
    )


if __name__ == "__main__":
    main()
