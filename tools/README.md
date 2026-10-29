# engine/tools — data bakers

Parameterised Node CLIs (Node >= 18, no npm deps except `compress-textures.mjs`) that produce the world data contracts MapWorld loads
(see ../../ENGINE.md, "Data contracts"). Every tool takes `--help`, `--dry-run` (prints the plan, makes no requests and
writes nothing), `--refresh` (ignore caches), `--out`, `--cache` and `--config <file.json>` (same keys as the flags;
flags win). Defaults are the Roadster San Francisco bake; each tool also exports an async function with the same
options in camelCase, which the game's thin wrappers in `../../tools/` call with explicit SF arguments.

| Tool | Output | Main flags (defaults) | Function |
|---|---|---|---|
| `bake-map.mjs` | city.json from OpenStreetMap (Overpass, OSM API fallback) | `--lat 37.792 --lon -122.3965 --size 2200 --outer 9000 --bridge-ext 3200,2200 --ua ...` | `bakeMap(opts)` |
| `bake-terrain.mjs` | terrain.json from AWS Terrarium tiles (OpenTopoData fallback) | `--city public/map/city.json [--lat --lon --outer] --fine -2600,-2900,3500,2500 --fine-cell 10 --fine-zoom 15 --outer-cell 40 --outer-zoom 13 --bay-min -2 --checks file.json` | `bakeTerrain(opts)` |
| `bake-farland.mjs` | farland.json (distant relief ring) | `--city ... [--lat --lon] --half 24000 --cell 250 --zoom 11 --sea -5` | `bakeFarland(opts)` |
| `fetch-textures.mjs` | PBR texture sets + manifest.json + CREDITS.md | `--sets textures.default.json --out public/textures --res 1k [--ktx2]` | `fetchTextures(opts)` |
| `compress-textures.mjs` | `<id>/*.ktx2` (Basis Universal: ETC1S albedo / rough / AO, UASTC + Zstd normals, mips, Y-flipped) + manifest `ktx2Maps`; copies three's Basis transcoder | `--dir public/textures --sets a,b --quality 160 --effort 2 --uastc-level 2 --rdo 0 --uastc id/map --etc1s map --jobs 8 --basis public/basis --basis-only --force` (Node >= 20.6; npm: `ktx2-encoder`, `jpeg-js`, `ktx-parse`) | `compressTextures(opts)`, `copyTranscoder(dest)` |
| `fetch-models.mjs` | sample character model (RobotExpressive.glb, CC0, three.js r180 examples) + CREDITS.md into `samples/character/models/` (gitignored) | `--out samples/character/models [--refresh] [--dry-run]` | `fetchModels(opts)` |

```bash
node engine/tools/bake-map.mjs --lat 40.7128 --lon -74.006 --size 1500 --bridge-ext 0,0 --out maps/nyc/city.json --dry-run
node engine/tools/bake-terrain.mjs --city maps/nyc/city.json --fine -1600,-1600,1600,1600 --out maps/nyc/terrain.json
```
`cli.mjs` holds the shared flag parser / usage printer.
