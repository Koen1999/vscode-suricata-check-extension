# Contributing

This extension is built based on the template provided by [Microsoft](https://github.com/microsoft/vscode-python-tools-extension-template).

## Linting and testing

1. `nox --session lint`
2. `nox --session tests`

## Update and Release build

1. Change `version` in `package.json`.
2. `nox --session update_packages`
3. `nox --session setup`
4. `nox --session build_package`
5. Upload the `.vsix` file to the [marketplace](https://marketplace.visualstudio.com/manage/publishers/koen1999).

Note that the above procedure is now automatically executed through a GitHub workflow.