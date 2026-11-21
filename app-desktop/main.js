/* NeuroFoil Flight Deck — Electron main process.
   Spawns the FastAPI sidecar on 127.0.0.1:8765, waits for health, then opens
   the app window pointed at that single local origin. Kills the sidecar on
   quit. No external network access. */
const { app, BrowserWindow, shell } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");

// The chrome-sandbox SUID helper is not configured on this machine; for a
// purely local single-window app that loads only 127.0.0.1 content, running
// without the Chromium sandbox is the accepted trade-off.
app.commandLine.appendSwitch("no-sandbox");

const PORT = 8765;
const BASE = `http://127.0.0.1:${PORT}`;
let sidecar = null;
let win = null;

function startSidecar() {
  const backendDir = path.join(__dirname, "backend");
  sidecar = spawn(process.execPath.includes("electron") ? "python3" : "python3",
    ["-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", String(PORT)],
    { cwd: backendDir, stdio: ["ignore", "pipe", "pipe"] });
  sidecar.stdout.on("data", (d) => process.stdout.write(`[sidecar] ${d}`));
  sidecar.stderr.on("data", (d) => process.stderr.write(`[sidecar] ${d}`));
  sidecar.on("exit", (code) => { sidecar = null; if (code && code !== 0) console.error(`sidecar exited ${code}`); });
}

function waitForHealth(retries = 60) {
  return new Promise((resolve, reject) => {
    const probe = (n) => {
      http.get(`${BASE}/api/health`, (res) => {
        res.resume();
        if (res.statusCode === 200) return resolve();
        retry(n);
      }).on("error", () => retry(n));
    };
    const retry = (n) => {
      if (n <= 0) return reject(new Error("sidecar did not become healthy"));
      setTimeout(() => probe(n - 1), 250);
    };
    probe(retries);
  });
}

function createWindow() {
  win = new BrowserWindow({
    width: 1440, height: 900, minWidth: 1100, minHeight: 700,
    backgroundColor: "#101214",
    title: "NeuroFoil — Flight Deck",
    autoHideMenuBar: true,
    webPreferences: {
      contextIsolation: true, nodeIntegration: false,
      preload: path.join(__dirname, "preload.js"),
    },
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  win.loadURL(BASE);
}

app.whenReady().then(async () => {
  startSidecar();
  try {
    await waitForHealth();
  } catch (err) {
    console.error(err);
    app.quit();
    return;
  }
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", () => {
  if (sidecar) {
    try { sidecar.kill(); } catch { /* already gone */ }
    sidecar = null;
  }
});

app.on("window-all-closed", () => {
  app.quit();
});
