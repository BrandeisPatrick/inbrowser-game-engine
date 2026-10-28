import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';

const here = (p) => fileURLToPath(new URL(p, import.meta.url));

// Dev-only second page: the procedural-city harness (samples/city/) at /city, served from outside the sample root
// through /@fs so the Canyon Run sample keeps the root URL; it is not part of `vite build`.
const cityPage = () => ({
  name: 'engine-city-page',
  configureServer(server) {
    server.middlewares.use(async (req, res, next) => {
      const u = req.url || '';
      if (!/^\/city\/?(\?.*)?$/.test(u)) return next();
      const html = fs.readFileSync(here('./samples/city/index.html'), 'utf8').replace(/__CITY_MAIN__/g, `/@fs/${here('./samples/city/main.js')}`);
      res.setHeader('Content-Type', 'text/html');
      res.end(await server.transformIndexHtml(u, html));
    });
  },
});

// Dev server / build for the samples. Samples import the engine as the package `@vibe/engine`; inside this
// repository that resolves to ./src (a game that installs the package gets it from node_modules instead).
export default defineConfig({
  root: here('./samples/canyon-run'),
  plugins: [cityPage()],
  server: { port: 5190, strictPort: true, fs: { allow: [here('.')] } },
  build: { target: 'es2022', outDir: here('./dist/canyon-run'), emptyOutDir: true },
  resolve: {
    alias: [
      { find: /^@vibe\/engine\/(core|physics|render|input|camera|audio|ui|world)$/, replacement: `${here('./src')}/$1/index.js` },
      { find: /^@vibe\/engine$/, replacement: here('./src/index.js') },
    ],
  },
});
