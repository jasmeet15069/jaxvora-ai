import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourceHtml = resolve(rootDir, "server", "index.html");
const sourceAssets = resolve(rootDir, "server", "assets");
const outputDir = resolve(rootDir, "frontend", "dist");
const outputHtml = resolve(outputDir, "index.html");

rmSync(outputDir, { recursive: true, force: true });
mkdirSync(outputDir, { recursive: true });

let html = readFileSync(sourceHtml, "utf8");
html = html.replace(
  /<title>.*?<\/title>/,
  "<title>Jaxvora - AI Command Center</title>",
);
html = html.replace(
  "</head>",
  '<meta name="description" content="Jaxvora autonomous multi-agent AI command center" />\n</head>',
);
html = html.replace(
  `  connectChatWs();
  connectLogsWs();
  connectAgentsWs();
  connectTasksWs();
  loadDashboard();
  loadAgents();

  // Poll dashboard every 30s
  setInterval(loadDashboard, 30000);
  setInterval(loadApprovals, 15000);`,
  `  const isVercelFrontend = location.hostname.endsWith('.vercel.app');
  if (!isVercelFrontend) {
    connectChatWs();
    connectLogsWs();
    connectAgentsWs();
    connectTasksWs();
  } else {
    document.getElementById('sys-status').textContent = 'online';
  }
  loadDashboard();
  loadAgents();

  // Vercel proxies HTTP to the VM backend; poll where WebSocket upgrades are unavailable.
  setInterval(loadDashboard, 30000);
  setInterval(loadAgents, 15000);
  setInterval(loadApprovals, 15000);`,
);

writeFileSync(outputHtml, html);
writeFileSync(resolve(outputDir, "robots.txt"), "User-agent: *\nAllow: /\n");
writeFileSync(
  resolve(outputDir, "_headers"),
  "/\n  Cache-Control: public, max-age=0, must-revalidate\n",
);
for (const assetName of ["favicon.ico", "favicon.png", "apple-touch-icon.png"]) {
  copyFileSync(resolve(sourceAssets, assetName), resolve(outputDir, assetName));
}

console.log(`Built Vercel frontend from ${sourceHtml}`);
