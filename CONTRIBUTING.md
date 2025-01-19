# Contributing

This extension is built based on the template provided by [Microsoft](https://github.com/microsoft/vscode-python-tools-extension-template).

## Update and Release build

1. `npm update`
2. `nox --session setup`
3. Change `version` in `package.json`.
4. `nox --session build_package`
5. Upload the `.vsix` file to the [marketplace](https://marketplace.visualstudio.com/manage/publishers/koen1999).
