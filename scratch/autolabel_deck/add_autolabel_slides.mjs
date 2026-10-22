import fs from "fs/promises";

const artifactPath =
  "/Users/ral3ply/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const {
  PresentationFile,
  row,
  column,
  grid,
  panel,
  text,
  shape,
  rule,
  fill,
  fixed,
  hug,
  wrap,
  fr,
} = await import(artifactPath);

const SRC = "scratch/autolabel_deck/source.pptx";
const OUT = "scratch/autolabel_deck/dsx_on_device_ai_labeling_proposal_autolabeling.pptx";
const PNG_DIR = "scratch/autolabel_deck/previews";

const C = {
  navy: "#0F172A",
  slate: "#334155",
  muted: "#64748B",
  line: "#E2E8F0",
  soft: "#F8FAFC",
  blue: "#2563EB",
  blue2: "#1D4ED8",
  white: "#FFFFFF",
};

const SLIDE = { width: 1280, height: 720 };
const titleStyle = { fontSize: 30, bold: true, color: C.navy };
const subtitleStyle = { fontSize: 16, color: C.muted };
const hStyle = { fontSize: 17, bold: true, color: C.navy };
const bodyStyle = { fontSize: 12.5, color: C.slate };
const smallStyle = { fontSize: 11, color: C.muted };
const chipStyle = { fontSize: 12.5, bold: true, color: C.blue2 };

function card(name, child, opts = {}) {
  return panel(
    {
      name,
      width: fill,
      height: fill,
      padding: { x: 18, y: 16 },
      fill: opts.fill ?? C.white,
      line: { color: C.line, width: 1 },
      borderRadius: 10,
      materialize: true,
    },
    child,
  );
}

function bullet(label, detail) {
  return row(
    { width: fill, height: hug, gap: 12, align: "start" },
    [
      shape({
        name: `dot-${label.replace(/\W+/g, "-").toLowerCase()}`,
        geometry: "ellipse",
        width: fixed(9),
        height: fixed(9),
        fill: C.blue,
      }),
      text([{ run: label, textStyle: { bold: true } }, { run: ` ${detail}` }], {
        width: fill,
        height: hug,
        style: bodyStyle,
      }),
    ],
  );
}

function step(num, title, detail) {
  return column(
    { width: fill, height: hug, gap: 5, align: "center" },
    [
      panel(
        {
          name: `step-${num}-circle`,
          width: fixed(38),
          height: fixed(38),
          fill: C.blue,
          borderRadius: "rounded-full",
          align: "center",
          justify: "center",
          materialize: true,
        },
        text(String(num), {
          width: fixed(34),
          height: hug,
          style: { fontSize: 15, bold: true, color: C.white, alignment: "center" },
        }),
      ),
      text(title, {
        width: fill,
        height: hug,
        style: { fontSize: 12, bold: true, color: C.navy, alignment: "center" },
      }),
      text(detail, {
        width: fill,
        height: hug,
        style: { fontSize: 9.2, color: C.slate, alignment: "center" },
      }),
    ],
  );
}

function addAutoLabelingSlide(presentation) {
  const { slide } = presentation.slides.insert({ after: presentation.slides.getItem(4) });
  slide.compose(
    column(
      { name: "root", width: fill, height: fill, padding: { x: 48, y: 38 }, gap: 22 },
      [
        column(
          { name: "title-stack", width: fill, height: hug, gap: 10 },
          [
            text("Auto-labeling work completed", {
              name: "slide-title",
              width: fill,
              height: hug,
              style: titleStyle,
            }),
            text(
              "The offline teacher path is no longer just a proposal: it now produces traceable segmentation labels, masks, overlays, and metadata from ADAS camera inputs.",
              {
                name: "slide-subtitle",
              width: wrap(1050),
                height: hug,
                style: subtitleStyle,
              },
            ),
          ],
        ),
        grid(
          {
            name: "main-grid",
            width: fill,
            height: fixed(470),
            columns: [fr(1.05), fr(0.95)],
            columnGap: 22,
          },
          [
            card(
              "pipeline-card",
              column(
                { width: fill, height: fill, gap: 18 },
                [
                  text("Implemented labeling pipeline", {
                    width: fill,
                    height: hug,
                    style: hStyle,
                  }),
                  row(
                    { width: fill, height: hug, gap: 9, align: "center" },
                    [
                      step("1", "Inputs", "Images and front/rear vehicle video clips"),
                      rule({ width: fixed(22), stroke: "#94A3B8", weight: 2 }),
                      step("2", "Qwen", "Dominant object prompts from vision-language inference"),
                      rule({ width: fixed(22), stroke: "#94A3B8", weight: 2 }),
                      step("3", "SAM3", "Masks propagated over frames"),
                      rule({ width: fixed(22), stroke: "#94A3B8", weight: 2 }),
                      step("4", "Export", "YOLO boxes, polygons, overlays"),
                    ],
                  ),
                  text(
                    "Designed for human-in-the-loop labeling: the model proposes labels, and every output remains inspectable and versioned.",
                    {
                      width: fill,
                      height: hug,
                      style: { fontSize: 13.5, color: C.slate },
                    },
                  ),
                ],
              ),
            ),
            column(
              { width: fill, height: fill, gap: 12 },
              [
                card(
                  "artifacts-card",
                  column(
                    { width: fill, height: fill, gap: 11 },
                    [
                      text("Artifacts now produced", { width: fill, height: hug, style: hStyle }),
                      bullet("YOLO segmentation labels", "labels/segments/*.txt with normalized mask polygons"),
                      bullet("QA visuals", "overlay images and preview videos for spot checks"),
                      bullet("Run metadata", "qwen_output.json, processing_log.json, and class mappings"),
                      bullet("Taxonomy tooling", "prompt mining, aliases, and canonical class renames"),
                    ],
                  ),
                ),
                panel(
                  {
                    name: "status-callout",
                    width: fill,
                    height: hug,
                    padding: { x: 16, y: 12 },
                    fill: "#EFF6FF",
                    borderRadius: 10,
                    materialize: true,
                  },
                  text("Status: ready to generate pseudo-labels for curated pilot data and feed YOLO fine-tuning.", {
                    width: fill,
                    height: hug,
                    style: chipStyle,
                  }),
                ),
              ],
            ),
          ],
        ),
      ],
    ),
    { frame: { left: 0, top: 0, width: SLIDE.width, height: SLIDE.height }, baseUnit: 8 },
  );
}

function addYoloSlide(presentation) {
  const { slide } = presentation.slides.insert({ after: presentation.slides.getItem(5) });
  slide.compose(
    column(
      { name: "root", width: fill, height: fill, padding: { x: 48, y: 38 }, gap: 20 },
      [
        column(
          { name: "title-stack", width: fill, height: hug, gap: 10 },
          [
            text("YOLO fine-tuning path is in place", {
              name: "slide-title",
              width: fill,
              height: hug,
              style: titleStyle,
            }),
            text(
              "Auto-labels are structured to become training data for a compact student segmenter that can later be exported and benchmarked on edge hardware.",
              {
                name: "slide-subtitle",
                width: wrap(1050),
                height: hug,
                style: subtitleStyle,
              },
            ),
          ],
        ),
        grid(
          {
            name: "training-grid",
            width: fill,
            height: fixed(415),
            columns: [fr(1), fr(1), fr(1)],
            columnGap: 16,
          },
          [
            card(
              "dataset-build-card",
              column(
                { width: fill, height: fill, gap: 12 },
                [
                  text("1. Dataset build", { width: fill, height: hug, style: hStyle }),
                  bullet("Discover pairs", "image paths from metadata plus segment label files"),
                  bullet("Split data", "train/validation layout with consistent stems"),
                  bullet("Write config", "Ultralytics data.yaml with nc and class names"),
                ],
              ),
            ),
            card(
              "training-card",
              column(
                { width: fill, height: fill, gap: 12 },
                [
                  text("2. Fine-tune YOLO-seg", { width: fill, height: hug, style: hStyle }),
                  bullet("Student model", "nano/small YOLO segmentation candidate"),
                  bullet("Train loop", "Ultralytics yolo segment train from generated labels"),
                  bullet("Device-aware", "CPU/CUDA/MPS selection and export-ready weights"),
                ],
              ),
              { fill: C.soft },
            ),
            card(
              "validation-card",
              column(
                { width: fill, height: fill, gap: 12 },
                [
                  text("3. Validate and iterate", { width: fill, height: hug, style: hStyle }),
                  bullet("Quality gates", "mAP, mask IoU, per-class precision/recall"),
                  bullet("Failure review", "scenario-level misses, false positives, drift"),
                  bullet("Next step", "edge export plus latency/FPS/memory benchmark"),
                ],
              ),
              { fill: C.soft },
            ),
          ],
        ),
        panel(
          {
            name: "bottom-callout",
            width: fill,
            height: hug,
            padding: { x: 18, y: 13 },
            fill: "#EFF6FF",
            borderRadius: 10,
            materialize: true,
          },
          text(
            "Current state: label generation, taxonomy cleanup, YOLO dataset assembly, and training notebooks are implemented; pilot runs should now focus on label quality thresholds and measurable edge benchmarks.",
            { width: fill, height: hug, style: { fontSize: 13.5, bold: true, color: C.blue2 } },
          ),
        ),
      ],
    ),
    { frame: { left: 0, top: 0, width: SLIDE.width, height: SLIDE.height }, baseUnit: 8 },
  );
}

async function writeBlob(blob, path) {
  if (blob?.data) {
    await fs.writeFile(path, Buffer.from(blob.data));
    return;
  }
  const ab = await blob.arrayBuffer();
  await fs.writeFile(path, Buffer.from(ab));
}

await fs.mkdir(PNG_DIR, { recursive: true });

const presentation = await PresentationFile.importPptx(await fs.readFile(SRC));
addAutoLabelingSlide(presentation);
addYoloSlide(presentation);

const pptx = await PresentationFile.exportPptx(presentation);
await writeBlob(pptx, OUT);

const reloaded = await PresentationFile.importPptx(await fs.readFile(OUT));
for (const slideIndex of [5, 6]) {
  const blob = await reloaded.export({
    slide: reloaded.slides.getItem(slideIndex),
    format: "png",
  });
  await writeBlob(blob, `${PNG_DIR}/slide-${slideIndex + 1}.png`);
}

console.log(JSON.stringify({ out: OUT, slideCount: reloaded.slides.count, previews: [`${PNG_DIR}/slide-6.png`, `${PNG_DIR}/slide-7.png`] }, null, 2));
