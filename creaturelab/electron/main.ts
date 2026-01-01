import 'dotenv/config'
import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import path from 'path'
import fs from 'fs'
import { TinkerAPI } from './api/tinker'
import { CheckpointManager } from './api/checkpoints'
import { NotesManager } from './api/notes'
import { ConversationsManager } from './api/conversations'

let mainWindow: BrowserWindow | null = null
let tinkerAPI: TinkerAPI | null = null
let checkpointManager: CheckpointManager | null = null
let notesManager: NotesManager | null = null
let conversationsManager: ConversationsManager | null = null

const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    backgroundColor: '#09090b',
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 15, y: 15 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173')
    mainWindow.webContents.openDevTools()
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

// Initialize API clients
function initializeServices() {
  const apiKey = process.env.TINKER_API_KEY
  if (!apiKey) {
    console.warn('TINKER_API_KEY not set - some features will be disabled')
  }

  tinkerAPI = new TinkerAPI(apiKey || '')
  checkpointManager = new CheckpointManager(tinkerAPI)

  // Notes stored in app data directory
  const notesPath = path.join(app.getPath('userData'), 'notes.json')
  notesManager = new NotesManager(notesPath)

  // Conversations stored in app data directory
  const conversationsPath = path.join(app.getPath('userData'), 'conversations.json')
  conversationsManager = new ConversationsManager(conversationsPath)
}

// IPC Handlers

// Checkpoint operations
ipcMain.handle('checkpoints:list', async () => {
  if (!checkpointManager) return { error: 'Not initialized' }
  return checkpointManager.listCheckpoints()
})

ipcMain.handle('checkpoints:getPersonas', async () => {
  if (!checkpointManager) return { error: 'Not initialized' }
  return checkpointManager.getPersonas()
})

// Chat operations
ipcMain.handle('chat:send', async (_event, { model, messages, options }) => {
  if (!tinkerAPI) return { error: 'Not initialized' }
  return tinkerAPI.chat(model, messages, options)
})

ipcMain.handle('chat:complete', async (_event, { model, prompt, options, requestId }) => {
  if (!tinkerAPI) return { error: 'Not initialized' }
  return tinkerAPI.complete(model, prompt, options, requestId)
})

ipcMain.handle('chat:abort', async (_event, requestId: string) => {
  if (!tinkerAPI) return false
  return tinkerAPI.abort(requestId)
})

// Notes operations
ipcMain.handle('notes:get', async (_event, checkpointId: string) => {
  if (!notesManager) return null
  return notesManager.get(checkpointId)
})

ipcMain.handle('notes:set', async (_event, checkpointId: string, notes: any) => {
  if (!notesManager) return { error: 'Not initialized' }
  return notesManager.set(checkpointId, notes)
})

ipcMain.handle('notes:getAll', async () => {
  if (!notesManager) return {}
  return notesManager.getAll()
})

// Conversations operations
ipcMain.handle('conversations:get', async (_event, checkpointId: string) => {
  if (!conversationsManager) return null
  return conversationsManager.get(checkpointId)
})

ipcMain.handle('conversations:set', async (_event, checkpointId: string, conversation: any) => {
  if (!conversationsManager) return { error: 'Not initialized' }
  return conversationsManager.set(checkpointId, conversation)
})

ipcMain.handle('conversations:getAll', async () => {
  if (!conversationsManager) return {}
  return conversationsManager.getAll()
})

ipcMain.handle('conversations:delete', async (_event, checkpointId: string) => {
  if (!conversationsManager) return
  return conversationsManager.delete(checkpointId)
})

// Export operations
ipcMain.handle('export:conversation', async (_event, { conversation, format }) => {
  if (!mainWindow) return { error: 'No window' }

  const { filePath } = await dialog.showSaveDialog(mainWindow, {
    title: 'Export Conversation',
    defaultPath: `conversation-${Date.now()}.${format}`,
    filters: [
      { name: format === 'json' ? 'JSON' : 'Markdown', extensions: [format] }
    ]
  })

  if (!filePath) return { cancelled: true }

  try {
    fs.writeFileSync(filePath, conversation, 'utf-8')
    return { success: true, path: filePath }
  } catch (err) {
    return { error: String(err) }
  }
})

// App status
ipcMain.handle('app:status', async () => {
  return {
    apiKeySet: !!process.env.TINKER_API_KEY,
    version: app.getVersion(),
  }
})

// App lifecycle
app.whenReady().then(() => {
  initializeServices()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
