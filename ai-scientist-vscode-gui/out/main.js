"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const child_process_1 = require("child_process");
const util_1 = require("util");
const readdir = (0, util_1.promisify)(fs.readdir);
const stat = (0, util_1.promisify)(fs.stat);
const readFile = (0, util_1.promisify)(fs.readFile);
const writeFile = (0, util_1.promisify)(fs.writeFile);
const unlink = (0, util_1.promisify)(fs.unlink);
let mainWindow = null;
let workflowProcess = null;
let agentProcess = null;
function createWindow() {
    mainWindow = new electron_1.BrowserWindow({
        width: 1600,
        height: 1000,
        minWidth: 1200,
        minHeight: 700,
        title: 'AI Scientist IDE',
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
        icon: path.join(__dirname, '../resources/icon.png'),
        backgroundColor: '#1e1e1e',
        frame: true,
        titleBarStyle: 'default',
    });
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'));
    // Open DevTools in development
    if (process.env.NODE_ENV === 'development') {
        mainWindow.webContents.openDevTools();
    }
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}
electron_1.app.on('ready', createWindow);
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        electron_1.app.quit();
    }
});
electron_1.app.on('activate', () => {
    if (mainWindow === null) {
        createWindow();
    }
});
// IPC Handlers for AI Scientist workflow
electron_1.ipcMain.handle('run-workflow', async (event, config) => {
    const sender = event.sender;
    if (workflowProcess) {
        return { success: false, error: 'Workflow already running' };
    }
    try {
        const { iterations, model, topic, outputPath } = config;
        const pythonPath = process.env.PYTHON || 'python'; // allow override
        const scriptPath = path.join(__dirname, '../../sciresearch_workflow.py');
        const args = [
            scriptPath,
            '--iterations', String(iterations),
            '--model', model,
            '--topic', topic,
            '--output', outputPath,
        ];
        workflowProcess = (0, child_process_1.spawn)(pythonPath, args, {
            cwd: path.join(__dirname, '../../'),
            env: { ...process.env },
            shell: process.platform === 'win32',
        });
        workflowProcess.stdout.on('data', (data) => {
            sender.send('workflow-log', data.toString());
        });
        workflowProcess.stderr.on('data', (data) => {
            sender.send('workflow-log', data.toString());
        });
        workflowProcess.on('close', (code) => {
            sender.send('workflow-exit', { code });
            workflowProcess = null;
        });
        workflowProcess.on('error', (err) => {
            sender.send('workflow-exit', { code: -1, error: err.message });
            workflowProcess = null;
        });
        return { success: true };
    }
    catch (error) {
        workflowProcess = null;
        return {
            success: false,
            error: error.message,
        };
    }
});
electron_1.ipcMain.handle('stop-workflow', async () => {
    if (!workflowProcess) {
        return { success: false, error: 'No workflow running' };
    }
    try {
        if (process.platform === 'win32') {
            (0, child_process_1.spawn)('taskkill', ['/pid', String(workflowProcess.pid), '/f', '/t']);
        }
        else {
            workflowProcess.kill('SIGTERM');
        }
        workflowProcess = null;
        return { success: true };
    }
    catch (error) {
        return { success: false, error: error.message };
    }
});
electron_1.ipcMain.handle('get-config', async () => {
    // Load and return current configuration
    try {
        const configPath = path.join(__dirname, '../../config.json');
        if (fs.existsSync(configPath)) {
            const configData = fs.readFileSync(configPath, 'utf-8');
            return JSON.parse(configData);
        }
        return null;
    }
    catch (error) {
        console.error('Failed to load config:', error);
        return null;
    }
});
electron_1.ipcMain.handle('save-config', async (event, config) => {
    try {
        const configPath = path.join(__dirname, '../../config.json');
        fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
        return { success: true };
    }
    catch (error) {
        return { success: false, error: error.message };
    }
});
// ============================================================================
// File System Handlers
// ============================================================================
// Open folder dialog
electron_1.ipcMain.handle('open-folder-dialog', async () => {
    const result = await electron_1.dialog.showOpenDialog(mainWindow, {
        properties: ['openDirectory'],
        title: 'Open Folder',
    });
    if (result.canceled || result.filePaths.length === 0) {
        return null;
    }
    return { path: result.filePaths[0] };
});
// Read directory recursively
async function readDirectoryRecursive(dirPath, depth = 0, maxDepth = 5) {
    if (depth > maxDepth)
        return [];
    const items = [];
    const entries = await readdir(dirPath, { withFileTypes: true });
    // Skip common non-essential directories
    const skipDirs = ['node_modules', '.git', '__pycache__', '.venv', 'venv', '.idea', '.vscode', 'dist', 'build', 'out'];
    for (const entry of entries) {
        if (entry.name.startsWith('.') && entry.name !== '.gitignore')
            continue;
        if (skipDirs.includes(entry.name))
            continue;
        const fullPath = path.join(dirPath, entry.name);
        if (entry.isDirectory()) {
            const children = await readDirectoryRecursive(fullPath, depth + 1, maxDepth);
            items.push({
                name: entry.name,
                path: fullPath,
                isDirectory: true,
                children,
            });
        }
        else {
            items.push({
                name: entry.name,
                path: fullPath,
                isDirectory: false,
            });
        }
    }
    return items;
}
electron_1.ipcMain.handle('read-directory', async (event, dirPath) => {
    try {
        const files = await readDirectoryRecursive(dirPath);
        return { files };
    }
    catch (error) {
        return { error: error.message };
    }
});
// Read file
electron_1.ipcMain.handle('read-file', async (event, filePath) => {
    try {
        const content = await readFile(filePath, 'utf-8');
        return { content };
    }
    catch (error) {
        return { error: error.message };
    }
});
// Write file
electron_1.ipcMain.handle('write-file', async (event, { path: filePath, content }) => {
    try {
        // Ensure directory exists
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        await writeFile(filePath, content, 'utf-8');
        return { success: true };
    }
    catch (error) {
        return { success: false, error: error.message };
    }
});
// Delete file
electron_1.ipcMain.handle('delete-file', async (event, filePath) => {
    try {
        await unlink(filePath);
        return { success: true };
    }
    catch (error) {
        return { success: false, error: error.message };
    }
});
// Search in files
electron_1.ipcMain.handle('search-files', async (event, { path: dirPath, query, caseSensitive, wholeWord, regex }) => {
    try {
        const results = [];
        async function searchInDir(searchPath) {
            const entries = await readdir(searchPath, { withFileTypes: true });
            const skipDirs = ['node_modules', '.git', '__pycache__', '.venv', 'venv'];
            for (const entry of entries) {
                if (skipDirs.includes(entry.name))
                    continue;
                const fullPath = path.join(searchPath, entry.name);
                if (entry.isDirectory()) {
                    await searchInDir(fullPath);
                }
                else {
                    // Skip binary files
                    const ext = path.extname(entry.name).toLowerCase();
                    const textExts = ['.txt', '.js', '.ts', '.py', '.json', '.html', '.css', '.md', '.tex', '.yml', '.yaml', '.xml', '.sh', '.bat', '.ps1'];
                    if (!textExts.includes(ext) && ext !== '')
                        continue;
                    try {
                        const content = await readFile(fullPath, 'utf-8');
                        const lines = content.split('\n');
                        let searchRegex;
                        if (regex) {
                            searchRegex = new RegExp(query, caseSensitive ? 'g' : 'gi');
                        }
                        else {
                            const escapedQuery = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                            const pattern = wholeWord ? `\\b${escapedQuery}\\b` : escapedQuery;
                            searchRegex = new RegExp(pattern, caseSensitive ? 'g' : 'gi');
                        }
                        lines.forEach((line, index) => {
                            if (searchRegex.test(line)) {
                                results.push({
                                    path: fullPath,
                                    line: index + 1,
                                    content: line,
                                });
                            }
                        });
                    }
                    catch (e) {
                        // Skip files that can't be read
                    }
                }
            }
        }
        await searchInDir(dirPath);
        return { results: results.slice(0, 100) }; // Limit results
    }
    catch (error) {
        return { error: error.message };
    }
});
// ============================================================================
// Settings
// ============================================================================
const settingsPath = path.join(electron_1.app.getPath('userData'), 'settings.json');
electron_1.ipcMain.handle('get-settings', async () => {
    try {
        if (fs.existsSync(settingsPath)) {
            const data = fs.readFileSync(settingsPath, 'utf-8');
            return JSON.parse(data);
        }
        return {};
    }
    catch (error) {
        return {};
    }
});
electron_1.ipcMain.handle('save-settings', async (event, settings) => {
    try {
        fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
        // Set environment variables for API keys
        if (settings.openaiKey) {
            process.env.OPENAI_API_KEY = settings.openaiKey;
        }
        if (settings.yunwuKey) {
            process.env.YUNWU_API_KEY = settings.yunwuKey;
        }
        return { success: true };
    }
    catch (error) {
        return { success: false, error: error.message };
    }
});
// ============================================================================
// AI Agent
// ============================================================================
electron_1.ipcMain.handle('agent-chat', async (event, { model, messages, context }) => {
    try {
        const pythonPath = process.env.PYTHON || 'python';
        const agentScriptPath = path.join(__dirname, '../../ai_agent.py');
        // Check if agent script exists, if not use a simpler approach
        if (!fs.existsSync(agentScriptPath)) {
            // Fallback: use the AI chat module directly
            const chatScriptPath = path.join(__dirname, '../../ai/chat.py');
            // Build prompt with context
            let systemPrompt = `You are an AI coding assistant (Copilot) that helps users modify files and write code.
You have access to the user's workspace and can suggest file modifications.

When you need to modify files, respond with JSON blocks like:
\`\`\`json
{"action": "modify", "path": "/path/to/file.py", "content": "new file content"}
\`\`\`

Current workspace: ${context.workspacePath || 'Not set'}
Open files: ${context.openFiles?.map((f) => f.path).join(', ') || 'None'}
`;
            if (context.activeFile) {
                systemPrompt += `\n\nCurrently editing: ${context.activeFile.path}\nContent:\n\`\`\`\n${context.activeFile.content}\n\`\`\``;
            }
            const fullMessages = [
                { role: 'system', content: systemPrompt },
                ...messages,
            ];
            // For now, return a simple response since we don't have direct API access from main process
            // In production, this would call the Python AI module
            return {
                response: `I understand your request. To fully implement the AI agent, we need to:\n\n1. Ensure the AI chat module is properly configured\n2. Set up API keys in Settings\n3. Use the workflow panel for research tasks\n\nYour request: "${messages[messages.length - 1]?.content || ''}"`,
                fileChanges: [],
            };
        }
        // Use the full agent script if available
        return new Promise((resolve, reject) => {
            const input = JSON.stringify({ model, messages, context });
            agentProcess = (0, child_process_1.spawn)(pythonPath, [agentScriptPath], {
                cwd: path.join(__dirname, '../../'),
                env: { ...process.env },
                shell: process.platform === 'win32',
            });
            let stdout = '';
            let stderr = '';
            agentProcess.stdin.write(input);
            agentProcess.stdin.end();
            agentProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            agentProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            agentProcess.on('close', (code) => {
                agentProcess = null;
                if (code === 0) {
                    try {
                        const result = JSON.parse(stdout);
                        resolve(result);
                    }
                    catch (e) {
                        resolve({ response: stdout, fileChanges: [] });
                    }
                }
                else {
                    resolve({ error: stderr || 'Agent process failed' });
                }
            });
            agentProcess.on('error', (err) => {
                agentProcess = null;
                resolve({ error: err.message });
            });
        });
    }
    catch (error) {
        return { error: error.message };
    }
});
//# sourceMappingURL=main.js.map