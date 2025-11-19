# Suricata Check

`suricata-check` is a command line utility to provide feedback on [Suricata](https://github.com/OISF/suricata) rules.
The tool can detect various issues including those covering syntax validity, interpretability, rule specificity, rule coverage, and efficiency.

## Features

`suricata-check`, offers the following features:

- Static analysis without Suricata installation for any operating system
- Detect using isssues pertaining to:
- - Missing mandatory options
- - Deviations from Suricata Style Guide best practices
- - - Missing/non-standard metadata fields, performance issues and more
- - Lack of rule coverage and specificity
- [Easily extendable with custom checkers](https://suricata-check.teuwen.net/checker.html)

For a complete overview, check out the [documentation](https://suricata-check.teuwen.net/).

## Configuration

### Option 1: `suricata-check.ini` configuration

You can create a project-wide configuration for `suricata-check` by creating a file called `suricata-check.ini` in your workspace.

The contents of `suricata-check.ini` can be configured as follows:
```ini
[suricata-check]
issue-severity="INFO"
include=["M.*", "S.*", "C.*"]
exclude=["S800"]
```

### Option 2: CLI Arguments

You can pass argument to the [`suricata-check` CLI](https://suricata-check.teuwen.net/cli_usage.html) using the `suricata-check.args` configuration option in VS Code.

For example, adding `"suricata-check.args": ["--issue-severity=WARNING"]` will only show issues with severity WARNING or greater.

It is also possible to enable or disable individual or groups of codes using the `--include` and `--exclude` options, which also accept regular expressions.

For example, the following configuration will include all issues concerning mandatory Suricata options and all issues based on the Suricata Style Guide, except S800 which prescribes `attack_target` as a mandatory metadata option:
```json
"suricata-check.args": [
  "--issue-severity=INFO",
  "--include=M.*",
  "--include=S.*",
  "--include=C.*",
  "--exclude=S800",
]
```

### Additional configuration options

For a complete overview of available command line options, check out the [CLI Reference](https://suricata-check.teuwen.net/cli.html).

## Suppressing issues for individual rules

You can suppress issues on a per-rule basis by adding a the `suricata-check` keyword to the `metadata` option.

For example, `metadata: suricata-check "C.*";` will disable all Community checkers for a the rule behind which the comment is placed.

More details can be found in the [documentation on suppressing rules](https://suricata-check.teuwen.net/ignore.html).

## Performance

For optimal performance, we suggest setting `suricata-check.importStrategy` to `fromEnvironment` and installing `suricata-check` with `regex` into your environment using `pip install -U suricata-check[performance]`. For more details check the [documentation page for the VSCode extension](https://suricata-check.teuwen.net/vscode.html).

## Alterative distributions

Suricata check is also available as a [command line tool](https://suricata-check.teuwen.net/cli_usage.html), which even offers [integration with CI/CD pipelines](https://suricata-check.teuwen.net/ci_cd.html).

When installed as a [PyPI Python package](https://pypi.org/project/suricata-check) you can also make use of [the API exposed by the module](https://suricata-check.teuwen.net/api_usage.html).

## Notes

[This repository](https://github.com/Koen1999/vscode-suricata-check-extension) only hosts the VS Code Extension comprising of the Language Server Protocol implementation for the tool to offer linting in Interactive Development Environments (IDE). You can find the main repository [here](https://github.com/Koen1999/suricata-check).

You can find the release version of this extension on the [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Koen1999.suricata-check).
