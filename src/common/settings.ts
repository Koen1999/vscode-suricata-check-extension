// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { ConfigurationChangeEvent, ConfigurationScope, WorkspaceConfiguration, WorkspaceFolder } from 'vscode';
import { getInterpreterDetails } from './python';
import { getConfiguration, getWorkspaceFolders } from './vscodeapi';

export interface ISettings {
    cwd: string;
    workspace: string;
    args: string[];
    path: string[];
    interpreter: string[];
    importStrategy: string;
    showNotifications: string;
}

export function getExtensionSettings(namespace: string, includeInterpreter?: boolean): Promise<ISettings[]> {
    return Promise.all(getWorkspaceFolders().map((w) => getWorkspaceSettings(namespace, w, includeInterpreter)));
}

function resolveVariables(value: string[], workspace?: WorkspaceFolder): string[] {
    const substitutions = new Map<string, string>();
    const home = process.env.HOME || process.env.USERPROFILE;
    if (home) {
        substitutions.set('${userHome}', home);
    }
    if (workspace) {
        substitutions.set('${workspaceFolder}', workspace.uri.fsPath);
    }
    substitutions.set('${cwd}', process.cwd());
    getWorkspaceFolders().forEach((w) => {
        substitutions.set('${workspaceFolder:' + w.name + '}', w.uri.fsPath);
    });

    return value.map((s) => {
        for (const [key, value] of substitutions) {
            s = s.replace(key, value);
        }
        return s;
    });
}

export function getInterpreterFromSetting(namespace: string, scope?: ConfigurationScope) {
    // When no scope is provided, explicitly pass `null` to indicate that we want
    // the effective value for any resource. This avoids the VS Code warning when
    // accessing a resource-scoped configuration without a resource.
    // `workspace.getConfiguration` accepts `null` at runtime but TypeScript's
    // type for ConfigurationScope doesn't include null. Cast to `any` to avoid
    // a type error while still passing `null` at runtime to get the effective value.
    const config = getConfiguration(namespace, (scope ?? null) as any);
    return config.get<string[]>('interpreter');
}

export async function getWorkspaceSettings(
    namespace: string,
    workspace: WorkspaceFolder,
    includeInterpreter?: boolean,
): Promise<ISettings> {
    const config = getConfiguration(namespace, workspace.uri);

    let interpreter: string[] = [];
    if (includeInterpreter) {
        interpreter = getInterpreterFromSetting(namespace, workspace) ?? [];
        if (interpreter.length === 0) {
            interpreter = (await getInterpreterDetails(workspace.uri)).path ?? [];
        }
    }

    const workspaceSetting = {
        cwd: workspace.uri.fsPath,
        workspace: workspace.uri.toString(),
        args: resolveVariables(config.get<string[]>(`args`) ?? [], workspace),
        path: resolveVariables(config.get<string[]>(`path`) ?? [], workspace),
        interpreter: resolveVariables(interpreter, workspace),
        importStrategy: config.get<string>(`importStrategy`) ?? 'useBundled',
        showNotifications: config.get<string>(`showNotifications`) ?? 'off',
    };
    return workspaceSetting;
}

function getGlobalValue<T>(config: WorkspaceConfiguration, key: string, defaultValue: T): T {
    const inspect = config.inspect<T>(key);
    return inspect?.globalValue ?? inspect?.defaultValue ?? defaultValue;
}

export async function getGlobalSettings(namespace: string, includeInterpreter?: boolean): Promise<ISettings> {
    // Explicitly pass `null` to getConfiguration to retrieve the effective
    // resource-scoped values for any resource without emitting a warning.
    // Pass `null as any` to getConfiguration so we retrieve the effective
    // resource-scoped values for any resource without a warning from VS Code.
    const config = getConfiguration(namespace, null as any);

    let interpreter: string[] = [];
    if (includeInterpreter) {
        interpreter = getGlobalValue<string[]>(config, 'interpreter', []);
        if (interpreter === undefined || interpreter.length === 0) {
            interpreter = (await getInterpreterDetails()).path ?? [];
        }
    }

    const setting = {
        cwd: process.cwd(),
        workspace: process.cwd(),
        args: getGlobalValue<string[]>(config, 'args', []),
        path: getGlobalValue<string[]>(config, 'path', []),
        interpreter: interpreter,
        importStrategy: getGlobalValue<string>(config, 'importStrategy', 'useBundled'),
        showNotifications: getGlobalValue<string>(config, 'showNotifications', 'off'),
    };
    return setting;
}

export function checkIfConfigurationChanged(e: ConfigurationChangeEvent, namespace: string): boolean {
    const settings = [
        `${namespace}.args`,
        `${namespace}.path`,
        `${namespace}.interpreter`,
        `${namespace}.importStrategy`,
        `${namespace}.showNotifications`,
    ];
    const changed = settings.map((s) => e.affectsConfiguration(s));
    return changed.includes(true);
}
