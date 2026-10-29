import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

// Dev server / build for the samples: a multi-page app rooted at samples/ (index.html lists them; each sample is a
// folder with its own index.html, served at /<name>/). Samples import the engine as the package `@vibe/engine`;
// inside this repository that resolves to ./src (a game that installs the package gets it from node_modules).
const SAMPLES = ['canyon-run', 'city', 'character', 'sandbox'].filter((s) => fs.existsSync(here(`./samples/${s}/index.html`)));
const input = { index: here('./samples/index.html') };
for (const s of SAMPLES) input[s] = here(`./samples/${s}/index.html`);

export default defineConfig({
  root: here('./samples'),
  server: { port: 5190, strictPort: true, fs: { allow: [here('.')] } },
  build: { target: 'es2022', outDir: here('./dist'), emptyOutDir: true, rollupOptions: { input } },
  resolve: {
    alias: [
      { find: /^@vibe\/engine\/(core|physics|render|anim|input|camera|audio|ui|world)$/, replacement: `${here('./src')}/$1/index.js` },
      { find: /^@vibe\/engine$/, replacement: here('./src/index.js') },
    ],
  },
});
