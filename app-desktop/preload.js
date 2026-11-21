/* Exposes nothing beyond the default sandboxed web view; the UI talks HTTP to
   the local sidecar only. Kept for future IPC (e.g., native save dialogs). */
window.addEventListener("DOMContentLoaded", () => {
  document.title = "NeuroFoil — Flight Deck";
});
