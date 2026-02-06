const { ipcRenderer } = require('electron');
const path = require('path');

// ============================================================================
// State Management
// ============================================================================
let workspacePath = null;
let fileTree = [];
let openTabs = [];
let activeTab = null;
let isRunning = false;
let agentConversation = [];

// ============================================================================
// DOM Elements
// ============================================================================
const elements = {
    // Workspace
    workspaceName: document.getElementById('workspace-name'),
    fileTreeEl: document.getElementById('file-tree'),
    
    // Activity Bar & Panels
    activityIcons: document.querySelectorAll('.activity-icon'),
    explorerPanel: document.getElementById('explorer-panel'),
    searchPanel: document.getElementById('search-panel'),
    agentPanel: document.getElementById('agent-panel'),
    workflowPanel: document.getElementById('workflow-panel'),
    settingsPanel: document.getElementById('settings-panel'),
    
    // Editor
    editorTabs: document.getElementById('editor-tabs'),
    editorContainer: document.getElementById('editor-container'),
    editorWelcome: document.getElementById('editor-welcome'),
    codeEditor: document.getElementById('code-editor'),
    
    // Bottom Panel
    panelTabs: document.querySelectorAll('.panel-tab'),
    terminalContent: document.getElementById('terminal-content'),
    outputContent: document.getElementById('output-content'),
    problemsContent: document.getElementById('problems-content'),
    consoleDiv: document.getElementById('console'),
    
    // Workflow
    iterationsInput: document.getElementById('iterations'),
    modelSelect: document.getElementById('model'),
    topicTextarea: document.getElementById('topic'),
    outputInput: document.getElementById('output'),
    runBtn: document.getElementById('run-btn'),
    stopBtn: document.getElementById('stop-btn'),
    statusSpan: document.getElementById('status'),
    currentIterationSpan: document.getElementById('current-iteration'),
    totalIterationsSpan: document.getElementById('total-iterations'),
    progressBar: document.getElementById('progress-bar'),
    
    // Agent
    agentChat: document.getElementById('agent-chat'),
    agentInput: document.getElementById('agent-input'),
    agentModel: document.getElementById('agent-model'),
    agentSendBtn: document.getElementById('agent-send-btn'),
    
    // Search
    searchInput: document.getElementById('search-input'),
    replaceInput: document.getElementById('replace-input'),
    searchBtn: document.getElementById('search-btn'),
    searchResults: document.getElementById('search-results'),
    
    // Settings
    openaiKey: document.getElementById('openai-key'),
    yunwuKey: document.getElementById('yunwu-key'),
    fontSize: document.getElementById('font-size'),
    theme: document.getElementById('theme'),
    saveSettingsBtn: document.getElementById('save-settings-btn'),
    
    // Status Bar
    statusBranch: document.getElementById('status-branch'),
    statusFiles: document.getElementById('status-files'),
    statusLine: document.getElementById('status-line'),
    statusLanguage: document.getElementById('status-language'),
    
    // Buttons
    openFolderBtn: document.getElementById('open-folder-btn'),
    openFolderBtnMain: document.getElementById('open-folder-btn-main'),
    quickOpenFolder: document.getElementById('quick-open-folder'),
    quickStartWorkflow: document.getElementById('quick-start-workflow'),
    newFileBtn: document.getElementById('new-file-btn'),
    refreshBtn: document.getElementById('refresh-btn'),
    clearConsoleBtn: document.getElementById('clear-console-btn'),
};

// ============================================================================
// Panel Management
// ============================================================================
function switchPanel(panelName) {
    // Update activity icons
    elements.activityIcons.forEach(icon => {
        icon.classList.toggle('active', icon.dataset.panel === panelName);
    });
    
    // Show/hide panels
    const panels = ['explorer', 'search', 'agent', 'workflow', 'settings'];
    panels.forEach(name => {
        const panel = document.getElementById(`${name}-panel`);
        if (panel) {
            panel.classList.toggle('hidden', name !== panelName);
        }
    });
}

function switchBottomTab(tabName) {
    elements.panelTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    elements.terminalContent.classList.toggle('hidden', tabName !== 'terminal');
    elements.outputContent.classList.toggle('hidden', tabName !== 'output');
    elements.problemsContent.classList.toggle('hidden', tabName !== 'problems');
}

// ============================================================================
// File Explorer
// ============================================================================
async function openFolder() {
    const result = await ipcRenderer.invoke('open-folder-dialog');
    if (result && result.path) {
        workspacePath = result.path;
        elements.workspaceName.textContent = path.basename(workspacePath);
        await refreshFileTree();
        addConsoleMessage(`Opened folder: ${workspacePath}`, 'info');
    }
}

async function refreshFileTree() {
    if (!workspacePath) return;
    
    const result = await ipcRenderer.invoke('read-directory', workspacePath);
    if (result && result.files) {
        fileTree = result.files;
        renderFileTree();
        elements.statusFiles.textContent = `${countFiles(fileTree)} files`;
    }
}

function countFiles(items) {
    let count = 0;
    for (const item of items) {
        if (item.isDirectory && item.children) {
            count += countFiles(item.children);
        } else if (!item.isDirectory) {
            count++;
        }
    }
    return count;
}

function renderFileTree(items = fileTree, container = elements.fileTreeEl, indent = 0) {
    if (indent === 0) {
        container.innerHTML = '';
        if (items.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No folder opened</p>
                    <button id="open-folder-btn-main" class="btn btn-primary">Open Folder</button>
                </div>
            `;
            document.getElementById('open-folder-btn-main')?.addEventListener('click', openFolder);
            return;
        }
    }
    
    // Sort: folders first, then files, alphabetically
    const sorted = [...items].sort((a, b) => {
        if (a.isDirectory && !b.isDirectory) return -1;
        if (!a.isDirectory && b.isDirectory) return 1;
        return a.name.localeCompare(b.name);
    });
    
    for (const item of sorted) {
        const div = document.createElement('div');
        div.className = `tree-item ${item.isDirectory ? 'folder' : 'file'}`;
        div.style.paddingLeft = `${10 + indent * 15}px`;
        
        const icon = item.isDirectory ? '📁' : getFileIcon(item.name);
        div.innerHTML = `<span class="icon">${icon}</span><span class="name">${item.name}</span>`;
        
        div.addEventListener('click', () => {
            if (item.isDirectory) {
                // Toggle folder expansion (simplified - just refresh for now)
            } else {
                openFile(item.path);
            }
        });
        
        container.appendChild(div);
        
        if (item.isDirectory && item.children) {
            renderFileTree(item.children, container, indent + 1);
        }
    }
}

function getFileIcon(filename) {
    const ext = path.extname(filename).toLowerCase();
    const icons = {
        '.js': '📜', '.ts': '📘', '.py': '🐍', '.json': '📋',
        '.html': '🌐', '.css': '🎨', '.md': '📝', '.tex': '📄',
        '.txt': '📃', '.yml': '⚙️', '.yaml': '⚙️', '.sh': '💻',
        '.bat': '💻', '.ps1': '💻', '.java': '☕', '.cpp': '⚙️',
        '.c': '⚙️', '.h': '⚙️', '.go': '🔵', '.rs': '🦀',
        '.rb': '💎', '.php': '🐘', '.sql': '🗃️', '.xml': '📰',
    };
    return icons[ext] || '📄';
}

// ============================================================================
// File Editor
// ============================================================================
async function openFile(filePath) {
    // Check if already open
    const existingTab = openTabs.find(t => t.path === filePath);
    if (existingTab) {
        activateTab(existingTab);
        return;
    }
    
    // Read file content
    const result = await ipcRenderer.invoke('read-file', filePath);
    if (!result || result.error) {
        addConsoleMessage(`Error opening file: ${result?.error || 'Unknown error'}`, 'error');
        return;
    }
    
    // Create new tab
    const tab = {
        path: filePath,
        name: path.basename(filePath),
        content: result.content,
        originalContent: result.content,
        modified: false,
    };
    
    openTabs.push(tab);
    renderTabs();
    activateTab(tab);
}

function renderTabs() {
    if (openTabs.length === 0) {
        elements.editorTabs.innerHTML = '<div class="tab-placeholder">Open a file to start editing</div>';
        return;
    }
    
    elements.editorTabs.innerHTML = openTabs.map(tab => `
        <div class="editor-tab ${tab === activeTab ? 'active' : ''} ${tab.modified ? 'modified' : ''}" data-path="${tab.path}">
            <span class="tab-icon">${getFileIcon(tab.name)}</span>
            <span class="tab-name">${tab.name}</span>
            <button class="close-tab" data-path="${tab.path}">×</button>
        </div>
    `).join('');
    
    // Add event listeners
    elements.editorTabs.querySelectorAll('.editor-tab').forEach(tabEl => {
        tabEl.addEventListener('click', (e) => {
            if (!e.target.classList.contains('close-tab')) {
                const tab = openTabs.find(t => t.path === tabEl.dataset.path);
                if (tab) activateTab(tab);
            }
        });
    });
    
    elements.editorTabs.querySelectorAll('.close-tab').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            closeTab(btn.dataset.path);
        });
    });
}

function activateTab(tab) {
    // Save current tab content
    if (activeTab) {
        activeTab.content = elements.codeEditor.value;
    }
    
    activeTab = tab;
    
    // Update UI
    elements.editorWelcome.classList.add('hidden');
    elements.codeEditor.classList.remove('hidden');
    elements.codeEditor.value = tab.content;
    
    // Update status bar
    updateStatusBar();
    
    renderTabs();
}

function closeTab(filePath) {
    const index = openTabs.findIndex(t => t.path === filePath);
    if (index === -1) return;
    
    const tab = openTabs[index];
    
    // TODO: Prompt to save if modified
    
    openTabs.splice(index, 1);
    
    if (activeTab === tab) {
        if (openTabs.length > 0) {
            activateTab(openTabs[Math.min(index, openTabs.length - 1)]);
        } else {
            activeTab = null;
            elements.editorWelcome.classList.remove('hidden');
            elements.codeEditor.classList.add('hidden');
        }
    }
    
    renderTabs();
}

async function saveCurrentFile() {
    if (!activeTab) return;
    
    const result = await ipcRenderer.invoke('write-file', {
        path: activeTab.path,
        content: elements.codeEditor.value,
    });
    
    if (result && result.success) {
        activeTab.content = elements.codeEditor.value;
        activeTab.originalContent = activeTab.content;
        activeTab.modified = false;
        renderTabs();
        addConsoleMessage(`Saved: ${activeTab.name}`, 'success');
    } else {
        addConsoleMessage(`Error saving file: ${result?.error || 'Unknown error'}`, 'error');
    }
}

function updateStatusBar() {
    if (!activeTab) {
        elements.statusLine.textContent = 'Ln 1, Col 1';
        elements.statusLanguage.textContent = 'Plain Text';
        return;
    }
    
    const ext = path.extname(activeTab.name).toLowerCase();
    const languages = {
        '.js': 'JavaScript', '.ts': 'TypeScript', '.py': 'Python',
        '.json': 'JSON', '.html': 'HTML', '.css': 'CSS',
        '.md': 'Markdown', '.tex': 'LaTeX', '.txt': 'Plain Text',
    };
    elements.statusLanguage.textContent = languages[ext] || 'Plain Text';
}

// ============================================================================
// AI Agent (Copilot)
// ============================================================================
async function sendAgentMessage() {
    const message = elements.agentInput.value.trim();
    if (!message) return;
    
    const model = elements.agentModel.value;
    
    // Add user message
    agentConversation.push({ role: 'user', content: message });
    renderAgentChat();
    elements.agentInput.value = '';
    
    // Prepare context
    const context = {
        workspacePath,
        openFiles: openTabs.map(t => ({ path: t.path, content: t.content })),
        activeFile: activeTab ? { path: activeTab.path, content: elements.codeEditor.value } : null,
    };
    
    // Add thinking indicator
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'agent-message assistant';
    thinkingDiv.innerHTML = '<div class="sender">🤖 AI Agent</div><div class="content">Thinking...</div>';
    elements.agentChat.appendChild(thinkingDiv);
    elements.agentChat.scrollTop = elements.agentChat.scrollHeight;
    
    try {
        const result = await ipcRenderer.invoke('agent-chat', {
            model,
            messages: agentConversation,
            context,
        });
        
        // Remove thinking indicator
        thinkingDiv.remove();
        
        if (result && result.response) {
            agentConversation.push({ role: 'assistant', content: result.response });
            
            // Check for file modifications
            if (result.fileChanges && result.fileChanges.length > 0) {
                for (const change of result.fileChanges) {
                    await applyFileChange(change);
                }
            }
        } else {
            agentConversation.push({ 
                role: 'assistant', 
                content: `Error: ${result?.error || 'Failed to get response'}` 
            });
        }
        
        renderAgentChat();
    } catch (error) {
        thinkingDiv.remove();
        agentConversation.push({ role: 'assistant', content: `Error: ${error.message}` });
        renderAgentChat();
    }
}

function renderAgentChat() {
    if (agentConversation.length === 0) {
        elements.agentChat.innerHTML = `
            <div class="agent-welcome">
                <h3>🤖 AI Copilot</h3>
                <p>I can help you with:</p>
                <ul>
                    <li>Modifying multiple files</li>
                    <li>Code generation</li>
                    <li>Refactoring</li>
                    <li>Bug fixing</li>
                    <li>Documentation</li>
                </ul>
            </div>
        `;
        return;
    }
    
    elements.agentChat.innerHTML = agentConversation.map(msg => `
        <div class="agent-message ${msg.role}">
            <div class="sender">${msg.role === 'user' ? '👤 You' : '🤖 AI Agent'}</div>
            <div class="content">${escapeHtml(msg.content)}</div>
        </div>
    `).join('');
    
    elements.agentChat.scrollTop = elements.agentChat.scrollHeight;
}

async function applyFileChange(change) {
    const { path: filePath, content, action } = change;
    
    if (action === 'create' || action === 'modify') {
        const result = await ipcRenderer.invoke('write-file', { path: filePath, content });
        if (result && result.success) {
            addConsoleMessage(`${action === 'create' ? 'Created' : 'Modified'}: ${filePath}`, 'success');
            
            // Update open tab if exists
            const tab = openTabs.find(t => t.path === filePath);
            if (tab) {
                tab.content = content;
                tab.originalContent = content;
                if (tab === activeTab) {
                    elements.codeEditor.value = content;
                }
            }
            
            await refreshFileTree();
        }
    } else if (action === 'delete') {
        const result = await ipcRenderer.invoke('delete-file', filePath);
        if (result && result.success) {
            addConsoleMessage(`Deleted: ${filePath}`, 'warning');
            closeTab(filePath);
            await refreshFileTree();
        }
    }
}

// ============================================================================
// Search
// ============================================================================
async function searchInFiles() {
    const query = elements.searchInput.value.trim();
    if (!query || !workspacePath) return;
    
    const result = await ipcRenderer.invoke('search-files', {
        path: workspacePath,
        query,
        caseSensitive: document.getElementById('case-sensitive')?.checked,
        wholeWord: document.getElementById('whole-word')?.checked,
        regex: document.getElementById('regex')?.checked,
    });
    
    if (result && result.results) {
        renderSearchResults(result.results);
    }
}

function renderSearchResults(results) {
    if (results.length === 0) {
        elements.searchResults.innerHTML = '<div class="empty-state"><p>No results found</p></div>';
        return;
    }
    
    elements.searchResults.innerHTML = results.map(r => `
        <div class="search-result" data-path="${r.path}" data-line="${r.line}">
            <span class="file-name">${path.basename(r.path)}</span>
            <span class="line-number">:${r.line}</span>
            <div class="line-content">${escapeHtml(r.content.trim())}</div>
        </div>
    `).join('');
    
    elements.searchResults.querySelectorAll('.search-result').forEach(el => {
        el.addEventListener('click', () => {
            openFile(el.dataset.path);
        });
    });
}

// ============================================================================
// Research Workflow
// ============================================================================
async function runWorkflow() {
    const iterations = parseInt(elements.iterationsInput.value);
    const model = elements.modelSelect.value;
    const topic = elements.topicTextarea.value.trim();
    const output = elements.outputInput.value.trim();
    
    if (!topic) {
        addConsoleMessage('Error: Please enter a research topic.', 'error');
        return;
    }
    
    if (iterations < 1 || iterations > 100) {
        addConsoleMessage('Error: Iterations must be between 1 and 100.', 'error');
        return;
    }
    
    // Update UI
    isRunning = true;
    elements.runBtn.disabled = true;
    elements.stopBtn.disabled = false;
    elements.statusSpan.textContent = 'Running';
    elements.statusSpan.style.color = '#4ec9b0';
    elements.totalIterationsSpan.textContent = iterations;
    elements.currentIterationSpan.textContent = '0';
    elements.progressBar.style.width = '0%';
    
    addConsoleMessage('Starting AI Scientist workflow...', 'info');
    addConsoleMessage(`Model: ${model}`, 'log');
    addConsoleMessage(`Iterations: ${iterations}`, 'log');
    addConsoleMessage(`Topic: ${topic}`, 'log');
    addConsoleMessage('─'.repeat(60), 'log');
    
    try {
        const result = await ipcRenderer.invoke('run-workflow', {
            iterations,
            model,
            topic,
            outputPath: output,
        });

        if (!result?.success) {
            addConsoleMessage(`Error: ${result?.error || 'unknown error'}`, 'error');
            elements.statusSpan.textContent = 'Error';
            elements.statusSpan.style.color = '#f48771';
            isRunning = false;
            elements.runBtn.disabled = false;
            elements.stopBtn.disabled = true;
        }
    } catch (error) {
        addConsoleMessage(`Error: ${error.message}`, 'error');
        elements.statusSpan.textContent = 'Error';
        elements.statusSpan.style.color = '#f48771';
        isRunning = false;
        elements.runBtn.disabled = false;
        elements.stopBtn.disabled = true;
    }
}

function stopWorkflow() {
    if (isRunning) {
        ipcRenderer.invoke('stop-workflow');
        elements.statusSpan.textContent = 'Stopped';
        elements.statusSpan.style.color = '#ce9178';
        addConsoleMessage('Stopping workflow...', 'warning');
    }
}

// ============================================================================
// Console
// ============================================================================
function addConsoleMessage(message, type = 'log') {
    const timestamp = new Date().toLocaleTimeString();
    const colors = {
        log: '#cccccc',
        info: '#4ec9b0',
        warning: '#ce9178',
        error: '#f48771',
        success: '#4ec9b0'
    };
    
    const color = colors[type] || colors.log;
    const line = document.createElement('div');
    line.style.color = color;
    line.style.marginBottom = '4px';
    line.textContent = `[${timestamp}] ${message}`;
    elements.consoleDiv.appendChild(line);
    elements.consoleDiv.scrollTop = elements.consoleDiv.scrollHeight;
}

function clearConsole() {
    elements.consoleDiv.innerHTML = '';
}

// ============================================================================
// Settings
// ============================================================================
async function loadSettings() {
    const settings = await ipcRenderer.invoke('get-settings');
    if (settings) {
        if (settings.openaiKey) elements.openaiKey.value = settings.openaiKey;
        if (settings.yunwuKey) elements.yunwuKey.value = settings.yunwuKey;
        if (settings.fontSize) elements.fontSize.value = settings.fontSize;
        if (settings.theme) elements.theme.value = settings.theme;
    }
}

async function saveSettings() {
    const settings = {
        openaiKey: elements.openaiKey.value,
        yunwuKey: elements.yunwuKey.value,
        fontSize: parseInt(elements.fontSize.value),
        theme: elements.theme.value,
    };
    
    const result = await ipcRenderer.invoke('save-settings', settings);
    if (result && result.success) {
        addConsoleMessage('Settings saved.', 'success');
        applySettings(settings);
    } else {
        addConsoleMessage(`Error saving settings: ${result?.error}`, 'error');
    }
}

function applySettings(settings) {
    if (settings.fontSize) {
        elements.codeEditor.style.fontSize = `${settings.fontSize}px`;
    }
}

// ============================================================================
// Utilities
// ============================================================================
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// IPC Listeners
// ============================================================================
ipcRenderer.on('workflow-log', (_event, message) => {
    addConsoleMessage(message.trimEnd(), 'log');
    
    // Try to parse iteration progress
    const match = message.match(/Iteration\s+(\d+)/i);
    if (match) {
        const current = parseInt(match[1]);
        elements.currentIterationSpan.textContent = current;
        const total = parseInt(elements.totalIterationsSpan.textContent) || 1;
        elements.progressBar.style.width = `${(current / total) * 100}%`;
    }
});

ipcRenderer.on('workflow-exit', (_event, payload) => {
    isRunning = false;
    elements.runBtn.disabled = false;
    elements.stopBtn.disabled = true;
    elements.progressBar.style.width = '100%';
    
    const code = payload?.code;
    if (code === 0) {
        elements.statusSpan.textContent = 'Completed';
        elements.statusSpan.style.color = '#4ec9b0';
        addConsoleMessage('Workflow completed successfully.', 'success');
    } else {
        elements.statusSpan.textContent = 'Error';
        elements.statusSpan.style.color = '#f48771';
        addConsoleMessage(`Workflow exited with code ${code ?? 'unknown'}`, 'error');
    }
});

// ============================================================================
// Event Listeners
// ============================================================================
function setupEventListeners() {
    // Activity Bar
    elements.activityIcons.forEach(icon => {
        icon.addEventListener('click', () => switchPanel(icon.dataset.panel));
    });
    
    // Bottom Panel Tabs
    elements.panelTabs.forEach(tab => {
        tab.addEventListener('click', () => switchBottomTab(tab.dataset.tab));
    });
    
    // File Explorer
    elements.openFolderBtn?.addEventListener('click', openFolder);
    elements.quickOpenFolder?.addEventListener('click', openFolder);
    elements.refreshBtn?.addEventListener('click', refreshFileTree);
    
    // Editor
    elements.codeEditor.addEventListener('input', () => {
        if (activeTab) {
            activeTab.modified = activeTab.content !== elements.codeEditor.value;
            renderTabs();
        }
    });
    
    elements.codeEditor.addEventListener('keyup', () => {
        const lines = elements.codeEditor.value.substr(0, elements.codeEditor.selectionStart).split('\n');
        const line = lines.length;
        const col = lines[lines.length - 1].length + 1;
        elements.statusLine.textContent = `Ln ${line}, Col ${col}`;
    });
    
    // Workflow
    elements.runBtn?.addEventListener('click', runWorkflow);
    elements.stopBtn?.addEventListener('click', stopWorkflow);
    elements.quickStartWorkflow?.addEventListener('click', () => switchPanel('workflow'));
    
    elements.iterationsInput?.addEventListener('input', () => {
        elements.totalIterationsSpan.textContent = elements.iterationsInput.value;
    });
    
    // Agent
    elements.agentSendBtn?.addEventListener('click', sendAgentMessage);
    elements.agentInput?.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            sendAgentMessage();
        }
    });
    
    // Search
    elements.searchBtn?.addEventListener('click', searchInFiles);
    elements.searchInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') searchInFiles();
    });
    
    // Settings
    elements.saveSettingsBtn?.addEventListener('click', saveSettings);
    
    // Console
    elements.clearConsoleBtn?.addEventListener('click', clearConsole);
    
    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            saveCurrentFile();
        } else if (e.ctrlKey && e.key === 'o') {
            e.preventDefault();
            openFolder();
        } else if (e.ctrlKey && e.shiftKey && e.key === 'F') {
            e.preventDefault();
            switchPanel('search');
        } else if (e.ctrlKey && e.shiftKey && e.key === 'A') {
            e.preventDefault();
            switchPanel('agent');
        }
    });
}

// ============================================================================
// Initialize
// ============================================================================
async function init() {
    setupEventListeners();
    await loadSettings();
    
    const config = await ipcRenderer.invoke('get-config');
    if (config) {
        if (config.default_model) elements.modelSelect.value = config.default_model;
        if (config.max_iterations) elements.iterationsInput.value = config.max_iterations;
    }
    
    addConsoleMessage('AI Scientist IDE initialized and ready.', 'info');
}

init();
