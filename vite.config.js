import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

// Dev server / build for the samples. Samples import the engine as the package `@vibe/engine`; inside this
// repository that resolves to ./src (a game that installs the package gets it from node_modules instead).
export default defineConfig({
  root: here('./samples/canyon-run'),
  server: { port: 5190, strictPort: true, fs: { allow: [here('.')] } },
  build: { target: 'es2022', outDir: here('./dist/canyon-run'), emptyOutDir: true },
  resolve: {
    alias: [
      { find: /^@vibe\/engine\/(core|physics|render|input|camera|audio|ui|world)$/, replacement: `${here('./src')}/$1/index.js` },
      { find: /^@vibe\/engine$/, replacement: here('./src/index.js') },
    ],
  },
});
