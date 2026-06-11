import { mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourceHtml = resolve(rootDir, "server", "index.html");
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

writeFileSync(outputHtml, html);
writeFileSync(resolve(outputDir, "robots.txt"), "User-agent: *\nAllow: /\n");
writeFileSync(
  resolve(outputDir, "_headers"),
  "/\n  Cache-Control: public, max-age=0, must-revalidate\n",
);

console.log(`Built Vercel frontend from ${sourceHtml}`);
