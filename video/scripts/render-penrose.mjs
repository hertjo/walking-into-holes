import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");
const sourceDir = path.join(root, "penrose");
const outputDir = path.join(root, "figures");
const stems = process.argv.length > 2 ? process.argv.slice(2) : ["teaser", "topology-vs-shape", "crossfit", "pipeline", "hero-mechanism"];

const dom = new JSDOM("<!doctype html><html><body></body></html>", {
  pretendToBeVisual: true,
});

Object.assign(global, {
  window: dom.window,
  document: dom.window.document,
  Node: dom.window.Node,
  DOMParser: dom.window.DOMParser,
  XMLSerializer: dom.window.XMLSerializer,
  HTMLElement: dom.window.HTMLElement,
  SVGElement: dom.window.SVGElement,
});

// jsdom deliberately omits a Canvas implementation. Penrose needs text metrics
// while compiling; the diagrams pin their geometry, so a deterministic metric
// approximation is sufficient for the plain-text labels. MathJax equations are
// still measured and rendered by Penrose itself.
Object.defineProperty(dom.window.HTMLCanvasElement.prototype, "getContext", {
  configurable: true,
  value() {
    return {
      font: "12px sans-serif",
      textBaseline: "alphabetic",
      measureText(text) {
        const match = /([0-9.]+)px/.exec(this.font);
        const px = match ? Number(match[1]) : 12;
        const width = 0.56 * px * [...text].length;
        return {
          width,
          actualBoundingBoxLeft: 0,
          actualBoundingBoxRight: width,
          actualBoundingBoxAscent: 0.74 * px,
          actualBoundingBoxDescent: 0.22 * px,
        };
      },
    };
  },
});

const { compile, optimize, showError, toSVG } = await import("@penrose/core");

await fs.mkdir(outputDir, { recursive: true });

for (const stem of stems) {
  const trio = {
    domain: await fs.readFile(path.join(sourceDir, `${stem}.domain`), "utf8"),
    substance: await fs.readFile(
      path.join(sourceDir, `${stem}.substance`),
      "utf8",
    ),
    style: await fs.readFile(path.join(sourceDir, `${stem}.style`), "utf8"),
    variation: "WalkingIntoHoles2026",
  };

  const compiled = await compile(trio);
  if (compiled.isErr()) {
    throw new Error(`${stem}: ${showError(compiled.error)}`);
  }

  const optimized = optimize(compiled.value);
  if (optimized.isErr()) {
    throw new Error(`${stem}: ${showError(optimized.error)}`);
  }

  const svg = await toSVG(
    optimized.value,
    async () => undefined,
    stem.replaceAll(/[^A-Za-z0-9_-]/g, "_"),
  );
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", "100%");

  await fs.writeFile(
    path.join(outputDir, `${stem}.svg`),
    `${svg.outerHTML}\n`,
    "utf8",
  );
  process.stdout.write(`rendered ${stem}\n`);
}
