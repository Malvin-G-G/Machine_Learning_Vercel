/* eslint-disable react-hooks/refs */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useRef, useState } from "react";
import cvModule from "@techstark/opencv-js";
import "./CamScannerFilter.css";

export type FilterMode =
  | "original"
  | "magic"
  | "magic_gray"
  | "grayscale"
  | "bw";
interface CamScannerFilterProps {
  image: string;
  onApply: (blob: Blob) => void;
}

/* ---------- OpenCV loader (works with old and new builds) ---------- */

/* ---------- OpenCV loader ---------- */
let cvPromise: Promise<{ cv: any }> | null = null;

function loadCv(): Promise<{ cv: any }> {
  if (cvPromise) return cvPromise;

  cvPromise = new Promise((resolve, reject) => {
    const mod: any = cvModule;
    const candidates: any[] = [mod];
    const isReady = (c: any) => !!c && typeof c.Mat === "function";

    // The package may be a thenable that resolves to the real instance
    if (typeof mod?.then === "function") {
      mod.then((resolved: any) => {
        if (resolved) candidates.push(resolved);
      });
    }

    // Older builds signal readiness through this callback
    try {
      mod.onRuntimeInitialized = () => candidates.push(mod);
    } catch {
      /* read-only, ignore */
    }

    const started = Date.now();
    const timer = setInterval(() => {
      const ready = candidates.find(isReady);
      if (ready) {
        clearInterval(timer);
        resolve({ cv: ready }); // wrapped so the Promise doesn't unwrap it
      } else if (Date.now() - started > 30000) {
        clearInterval(timer);
        cvPromise = null;
        reject(new Error("OpenCV failed to initialize within 30s"));
      }
    }, 50);
  });

  return cvPromise;
}

/* ---------- Helpers ---------- */

// Replacement for cv.imread: canvas -> RGBA Mat
function readMat(cv: any, canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d")!;
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const mat = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC4);
  mat.data.set(data.data);
  return mat;
}

// Replacement for cv.imshow: Mat (1, 3 or 4 channels) -> canvas
function showMat(cv: any, mat: any, canvas: HTMLCanvasElement) {
  const rgba = new cv.Mat();
  const channels = mat.channels();

  if (channels === 1) cv.cvtColor(mat, rgba, cv.COLOR_GRAY2RGBA);
  else if (channels === 3) cv.cvtColor(mat, rgba, cv.COLOR_RGB2RGBA);
  else mat.copyTo(rgba);

  if (canvas.width !== rgba.cols) canvas.width = rgba.cols;
  if (canvas.height !== rgba.rows) canvas.height = rgba.rows;
  canvas
    .getContext("2d")!
    .putImageData(
      new ImageData(new Uint8ClampedArray(rgba.data), rgba.cols, rgba.rows),
      0,
      0,
    );

  rgba.delete();
}

// Estimate the paper background of a single 8-bit channel and divide it out.
// This removes shadows / uneven lighting and turns the paper white.
function normalizeChannel(cv: any, channel: any): any {
  const scale = Math.min(1, 600 / Math.max(channel.cols, channel.rows));
  const small = new cv.Mat();
  const bgSmall = new cv.Mat();
  const bg = new cv.Mat();
  const out = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(7, 7));

  cv.resize(
    channel,
    small,
    new cv.Size(
      Math.round(channel.cols * scale),
      Math.round(channel.rows * scale),
    ),
    0,
    0,
    cv.INTER_AREA,
  );

  cv.dilate(small, bgSmall, kernel); // erase dark text/ink
  cv.medianBlur(bgSmall, bgSmall, 21); // smooth lighting estimate
  cv.resize(
    bgSmall,
    bg,
    new cv.Size(channel.cols, channel.rows),
    0,
    0,
    cv.INTER_LINEAR,
  );

  cv.divide(channel, bg, out, 255, -1);

  small.delete();
  bgSmall.delete();
  bg.delete();
  kernel.delete();
  return out;
}

function unsharp(cv: any, mat: any, amount = 0.6) {
  const blur = new cv.Mat();
  cv.GaussianBlur(mat, blur, new cv.Size(0, 0), 2);
  cv.addWeighted(mat, 1 + amount, blur, -amount, 0, mat);
  blur.delete();
}

/* ---------- Two-stage pipeline: heavy stage is cached, slider stage is cheap ---------- */

const PREVIEW_MAX = 480; // live preview size (small = fast)
const APPLY_MAX = 1600; // final output size (model only needs 224px)

type BaseKind = "rgb" | "color" | "gray";

const kindOf = (m: FilterMode): BaseKind =>
  m === "original" ? "rgb" : m === "magic" || m === "magic_gray" ? "color" : "gray";

function scaleCanvas(src: HTMLCanvasElement, max: number) {
  const k = Math.min(1, max / Math.max(src.width, src.height));
  if (k === 1) return src;
  const c = document.createElement("canvas");
  c.width = Math.round(src.width * k);
  c.height = Math.round(src.height * k);
  const ctx = c.getContext("2d")!;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, 0, 0, c.width, c.height);
  return c;
}

// Heavy stage (shadow/background normalization). Independent of sliders -> cached.
function buildBase(cv: any, source: HTMLCanvasElement, kind: BaseKind): any {
  const src = readMat(cv, source);
  const rgb = new cv.Mat();
  cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
  src.delete();
  if (kind === "rgb") return rgb;

  if (kind === "gray") {
    const gray = new cv.Mat();
    cv.cvtColor(rgb, gray, cv.COLOR_RGB2GRAY);
    rgb.delete();
    const norm = normalizeChannel(cv, gray);
    gray.delete();
    return norm;
  }

  const channels = new cv.MatVector();
  cv.split(rgb, channels);
  const normalized = new cv.MatVector();
  for (let i = 0; i < 3; i++) {
    const ch = channels.get(i);
    normalized.push_back(normalizeChannel(cv, ch));
    ch.delete();
  }
  const merged = new cv.Mat();
  cv.merge(normalized, merged);
  channels.delete();
  normalized.delete();
  rgb.delete();
  return merged;
}

// Cheap stage (runs on every slider tick). Caller deletes the result.
function finish(cv: any, base: any, mode: FilterMode, contrast: number, threshold: number): any {
  const out = new cv.Mat();
  if (mode === "original") {
    base.copyTo(out);
    return out;
  }
  if (mode === "bw") {
    cv.threshold(base, out, threshold, 255, cv.THRESH_BINARY);
    return out;
  }
  const boosted = new cv.Mat();
  base.convertTo(boosted, -1, contrast, -(contrast - 1) * 110);
  unsharp(cv, boosted, 0.6);
  if (mode === "magic_gray") {
    cv.cvtColor(boosted, out, cv.COLOR_RGB2GRAY);
    boosted.delete();
    return out;
  }
  out.delete();
  return boosted;
}

/* ---------- Component ---------- */

export default function CamScannerFilter({
  image,
  onApply,
}: CamScannerFilterProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sourceRef = useRef<HTMLCanvasElement | null>(null);
  const previewSrcRef = useRef<HTMLCanvasElement | null>(null);
  const baseRef = useRef<Partial<Record<BaseKind, any>>>({});
  const [applying, setApplying] = useState(false);

  function clearBases() {
    Object.values(baseRef.current).forEach((m) => {
      try {
        m.delete();
      } catch {
        /* already freed */
      }
    });
    baseRef.current = {};
  }

  // Free cached OpenCV memory on unmount
  useEffect(() => clearBases, []);

  const [cvReady, setCvReady] = useState(false);
  const [imgReady, setImgReady] = useState(false);
  const [mode, setMode] = useState<FilterMode>("magic_gray");
  const [contrast, setContrast] = useState(1.25);
  const [threshold, setThreshold] = useState(185);
  const cvRef = useRef<any>(null); // <-- add this

  // Load OpenCV
  useEffect(() => {
    let alive = true;
    loadCv()
      .then(({ cv }) => {
        if (!alive) return;
        cvRef.current = cv;
        setCvReady(true);
      })
      .catch((err) => console.error(err));
    return () => {
      alive = false;
    };
  }, []);

  // Load image into an offscreen source canvas
  useEffect(() => {
    if (!image) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setImgReady(false);
    const img = new Image();
    img.onload = () => {
      const src = document.createElement("canvas");
      src.width = img.naturalWidth;
      src.height = img.naturalHeight;
      src.getContext("2d")!.drawImage(img, 0, 0);
      sourceRef.current = src;
      previewSrcRef.current = scaleCanvas(src, PREVIEW_MAX);
      clearBases();
      setImgReady(true);
    };
    img.onerror = () => console.error("Failed to load filter image:", image);
    img.src = image;
    return () => {
      img.onload = null;
      img.onerror = null;
    };
  }, [image]);

  // Live preview: small image + cached heavy stage + one render per frame
  useEffect(() => {
    const canvas = canvasRef.current;
    const preview = previewSrcRef.current;
    const cv = cvRef.current;
    if (!cvReady || !imgReady || !canvas || !preview || !cv) return;

    const id = requestAnimationFrame(() => {
      let result: any;
      try {
        const kind = kindOf(mode);
        if (!baseRef.current[kind]) baseRef.current[kind] = buildBase(cv, preview, kind);
        result = finish(cv, baseRef.current[kind], mode, contrast, threshold);
        showMat(cv, result, canvas);
      } catch (err) {
        console.error("OpenCV filter failed:", err);
      } finally {
        try {
          result?.delete();
        } catch {
          /* already freed */
        }
      }
    });

    return () => cancelAnimationFrame(id);
  }, [cvReady, imgReady, mode, contrast, threshold]);

  // Final output: run the full pipeline once at higher resolution
  function applyFilter() {
    const cv = cvRef.current;
    const source = sourceRef.current;
    if (!cv || !source || applying) return;

    setApplying(true);
    setTimeout(() => {
      let base: any;
      let result: any;
      try {
        base = buildBase(cv, scaleCanvas(source, APPLY_MAX), kindOf(mode));
        result = finish(cv, base, mode, contrast, threshold);
        const out = document.createElement("canvas");
        showMat(cv, result, out);
        out.toBlob(
          (blob) => {
            setApplying(false);
            if (blob) onApply(blob);
          },
          "image/jpeg",
          0.95,
        );
      } catch (err) {
        console.error("Apply failed:", err);
        setApplying(false);
      } finally {
        try {
          base?.delete();
          result?.delete();
        } catch {
          /* already freed */
        }
      }
    }, 30);
  }

  return (
    <section className='scanner-filter'>
      {/* Header */}
      <header className='scanner-header'>
        <div>
          <span className='scanner-eyebrow'>IMAGE EDITOR</span>
          <h2>Enhance your document</h2>
          <p>Adjust your image to get a clean, readable scan.</p>
        </div>

        <span
          className={`scanner-status ${cvReady && imgReady ? "ready" : ""}`}
        >
          <span className='status-dot' />
          {cvReady && imgReady ? "Ready to edit" : "Preparing scanner..."}
        </span>
      </header>

      {/* Image preview */}
      <div className='scanner-preview'>
        <div className='preview-toolbar'>
          <span>
            <span className='preview-icon'>▧</span>
            Preview
          </span>
          <span className='preview-resolution'>
            {sourceRef.current
              ? `${sourceRef.current.width} × ${sourceRef.current.height}`
              : "Loading image"}
          </span>
        </div>

        <div className='preview-image'>
          {!cvReady || !imgReady ? (
            <div className='preview-loading'>
              <span className='loading-spinner' />
              <strong>Preparing your image</strong>
              <p>Initializing image processing engine...</p>
            </div>
          ) : null}

          <canvas
            ref={canvasRef}
            className='scanner-canvas'
            style={{
              display: cvReady && imgReady ? "block" : "none",
            }}
          />
        </div>

        <div className='preview-footer'>
          <span>LIVE PREVIEW</span>
          <span>Changes are applied automatically</span>
        </div>
      </div>

      {/* Filter selection */}
      <div className='scanner-section'>
        <div className='section-heading'>
          <div>
            <h3>Choose a filter</h3>
            <p>Select the appearance that works best for your document.</p>
          </div>
          <span className='section-count'>05 FILTERS</span>
        </div>

        <div className='filter-grid'>
          {[
            {
              id: "original",
              icon: "◉",
              title: "Original",
              description: "No adjustments",
            },
            {
              id: "magic",
              icon: "✦",
              title: "Magic Color",
              description: "Enhance colors",
            },
            {
              id: "magic_gray",
              icon: "◐",
              title: "Magic Gray",
              description: "Clean grayscale",
            },
            {
              id: "grayscale",
              icon: "◑",
              title: "Grayscale",
              description: "Natural gray tones",
            },
            {
              id: "bw",
              icon: "◧",
              title: "Black & White",
              description: "High contrast scan",
            },
          ].map((filter) => (
            <button
              key={filter.id}
              type='button'
              className={`filter-card ${mode === filter.id ? "active" : ""}`}
              onClick={() => setMode(filter.id as FilterMode)}
              aria-pressed={mode === filter.id}
            >
              <span className='filter-icon'>{filter.icon}</span>
              <span className='filter-name'>{filter.title}</span>
              <span className='filter-description'>{filter.description}</span>
              {mode === filter.id && <span className='filter-check'>✓</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Adjustment panel */}
      {(mode === "magic" ||
        mode === "magic_gray" ||
        mode === "grayscale" ||
        mode === "bw") && (
        <div className='scanner-section adjustment-panel'>
          <div className='section-heading'>
            <div>
              <h3>Fine adjustments</h3>
              <p>Fine-tune your scan to achieve the desired result.</p>
            </div>
            <button
              type='button'
              className='reset-button'
              onClick={() => {
                setContrast(1.25);
                setThreshold(185);
              }}
            >
              ↺ Reset
            </button>
          </div>

          {mode !== "bw" ? (
            <div className='adjustment-control'>
              <div className='adjustment-label'>
                <label htmlFor='scanner-contrast'>Contrast</label>
                <output>{contrast.toFixed(2)}×</output>
              </div>

              <input
                id='scanner-contrast'
                className='scanner-range'
                type='range'
                min='1'
                max='2'
                step='0.01'
                value={contrast}
                onChange={(e) => setContrast(Number(e.target.value))}
                style={
                  {
                    "--range-progress": `${((contrast - 1) / 1) * 100}%`,
                  } as React.CSSProperties
                }
              />

              <div className='range-labels'>
                <span>Low</span>
                <span>High</span>
              </div>
            </div>
          ) : (
            <div className='adjustment-control'>
              <div className='adjustment-label'>
                <label htmlFor='scanner-threshold'>Threshold</label>
                <output>{threshold}</output>
              </div>

              <input
                id='scanner-threshold'
                className='scanner-range'
                type='range'
                min='100'
                max='240'
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                style={
                  {
                    "--range-progress": `${((threshold - 100) / 140) * 100}%`,
                  } as React.CSSProperties
                }
              />

              <div className='range-labels'>
                <span>Darker</span>
                <span>Lighter</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Footer actions */}
      <footer className='scanner-actions'>
        <div className='action-description'>
          <span className='action-check'>✓</span>
          <span>
            <strong>Ready to apply?</strong>
            <small>Your original image remains unchanged.</small>
          </span>
        </div>

        <button
          type='button'
          className='apply-button'
          onClick={applyFilter}
          disabled={!cvReady || !imgReady || applying}
        >
          {applying ? "Processing..." : "Apply Filter"}
          <span>→</span>
        </button>
      </footer>
    </section>
  );
}