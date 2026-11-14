/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import cvModule from "@techstark/opencv-js";

import Camera from "./Camera";
import ImageCropper from "./ImageCropper";
import type { PredictionResult } from "./PredictImage";

interface InkScannerProps {
  classNames: string[];
}

type RunnerMode = "local" | "remote";
type ModelOption = "ensemble" | "efficient" | "cnn";

const V2_MODEL_CONFIGS = [
  {
    key: "cnn",
    label: "CNN (Multi-Line)",
    primaryUrl: "/cnn_model_v2/model.json",
    fallbackUrl: "/cnn_model_web/model.json",
  },
  {
    key: "efficient",
    label: "EfficientNetB0 (Fine-Tuned)",
    primaryUrl: "/efficient_model_v2/model.json",
    fallbackUrl: "/efficient_model_web/model.json",
  },
] as const;

type V2ModelKey = (typeof V2_MODEL_CONFIGS)[number]["key"];
type V2Models = Partial<Record<V2ModelKey, tf.GraphModel>>;

let cvInstancePromise: Promise<{ cv: any }> | null = null;

function initOpenCv(): Promise<{ cv: any }> {
  if (cvInstancePromise) return cvInstancePromise;

  cvInstancePromise = new Promise((resolve, reject) => {
    const mod: any = cvModule;
    const candidates: any[] = [mod];
    const isReady = (c: any) => !!c && typeof c.Mat === "function";

    if (typeof mod?.then === "function") {
      mod.then((resolved: any) => {
        if (resolved) candidates.push(resolved);
      });
    }

    try {
      mod.onRuntimeInitialized = () => candidates.push(mod);
    } catch {
      /* read-only */
    }

    const started = Date.now();
    const timer = setInterval(() => {
      const ready = candidates.find(isReady);
      if (ready) {
        clearInterval(timer);
        resolve({ cv: ready });
      } else if (Date.now() - started > 30000) {
        clearInterval(timer);
        cvInstancePromise = null;
        reject(new Error("OpenCV failed to initialize within 30s"));
      }
    }, 50);
  });

  return cvInstancePromise;
}

// Ink-centric preprocessing with multi-line strip stacking (224x224)
function buildInkCanvas(
  cv: any,
  sourceCanvas: HTMLCanvasElement,
  targetSize = 224,
  trimBorder = false,
): HTMLCanvasElement {
  const ctx = sourceCanvas.getContext("2d")!;
  const imgData = ctx.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  const srcMat = new cv.Mat(sourceCanvas.height, sourceCanvas.width, cv.CV_8UC4);
  srcMat.data.set(imgData.data);

  let gray = new cv.Mat();
  cv.cvtColor(srcMat, gray, cv.COLOR_RGBA2GRAY);
  srcMat.delete();

  // Optional auto-trim for dark camera borders
  if (trimBorder) {
    const hRaw = gray.rows;
    const wRaw = gray.cols;
    const paperMask = new cv.Mat();
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();

    cv.threshold(gray, paperMask, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    cv.findContours(paperMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let maxArea = 0;
    let bestRect: any = null;
    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area > maxArea) {
        maxArea = area;
        bestRect = cv.boundingRect(cnt);
      }
      cnt.delete();
    }

    if (bestRect && maxArea > 0.25 * (hRaw * wRaw)) {
      const marginY = Math.floor(bestRect.height * 0.03);
      const marginX = Math.floor(bestRect.width * 0.02);
      const newH = bestRect.height - 2 * marginY;
      const newW = bestRect.width - 2 * marginX;
      if (newH > 20 && newW > 50) {
        const roi = gray.roi(
          new cv.Rect(bestRect.x + marginX, bestRect.y + marginY, newW, newH),
        );
        const croppedPaper = new cv.Mat();
        roi.copyTo(croppedPaper);
        roi.delete();
        gray.delete();
        gray = croppedPaper;
      }
    }

    paperMask.delete();
    contours.delete();
    hierarchy.delete();
  }

  // 1. Upscale low-height images (< 120px)
  const h0 = gray.rows;
  const w0 = gray.cols;
  if (h0 < 120) {
    const scaleUp = 150.0 / Math.max(h0, 1);
    const upMat = new cv.Mat();
    cv.resize(
      gray,
      upMat,
      new cv.Size(Math.floor(w0 * scaleUp), 150),
      0,
      0,
      cv.INTER_CUBIC,
    );
    gray.delete();
    gray = upMat;
  }

  // 2. Background estimation using Morphological Closing + Gaussian Blur
  const closeKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(35, 35));
  const bgClosed = new cv.Mat();
  const bgSmooth = new cv.Mat();
  const normPaper = new cv.Mat();

  cv.morphologyEx(gray, bgClosed, cv.MORPH_CLOSE, closeKernel);
  cv.GaussianBlur(bgClosed, bgSmooth, new cv.Size(51, 51), 0, 0);
  cv.divide(gray, bgSmooth, normPaper, 255, -1);

  closeKernel.delete();
  bgClosed.delete();
  bgSmooth.delete();
  gray.delete();

  // 3. Polarity inversion (background = 0, ink = 90..255)
  const rows = normPaper.rows;
  const cols = normPaper.cols;
  const totalPixels = rows * cols;

  const inkMat = new cv.Mat(rows, cols, cv.CV_8UC1);
  for (let i = 0; i < totalPixels; i++) {
    inkMat.data[i] = 255 - normPaper.data[i];
  }
  normPaper.delete();

  const dummyMask = new cv.Mat();
  const otsuThresh = cv.threshold(
    inkMat,
    dummyMask,
    0,
    255,
    cv.THRESH_BINARY + cv.THRESH_OTSU,
  );
  dummyMask.delete();

  const noiseFloor = Math.max(25, Math.floor(otsuThresh * 0.75));
  const inkClean = new Uint8Array(totalPixels);
  let maxClean = 0;
  let minClean = 255;

  for (let i = 0; i < totalPixels; i++) {
    const val = inkMat.data[i] >= noiseFloor ? inkMat.data[i] : 0;
    inkClean[i] = val;
    if (val > maxClean) maxClean = val;
    if (val < minClean) minClean = val;
  }
  inkMat.delete();

  if (maxClean > 0) {
    const range = Math.max(1, maxClean - minClean);
    for (let i = 0; i < totalPixels; i++) {
      if (inkClean[i] > 0) {
        const stretched = ((inkClean[i] - minClean) * 255.0) / range;
        const boosted = Math.round(stretched * 1.2 + 50);
        inkClean[i] = Math.min(255, Math.max(90, boosted));
      }
    }
  }

  // 4. Tight bounding-box crop around ink strokes
  const binMat = new cv.Mat(rows, cols, cv.CV_8UC1);
  for (let i = 0; i < totalPixels; i++) {
    binMat.data[i] = inkClean[i] > 0 ? 255 : 0;
  }

  const connectKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(15, 5));
  const connected = new cv.Mat();
  cv.dilate(binMat, connected, connectKernel, new cv.Point(-1, -1), 2);
  binMat.delete();
  connectKernel.delete();

  let minX = cols;
  let minY = rows;
  let maxX = -1;
  let maxY = -1;

  for (let r = 0; r < rows; r++) {
    const rowOffset = r * cols;
    for (let c = 0; c < cols; c++) {
      if (connected.data[rowOffset + c] > 0) {
        if (c < minX) minX = c;
        if (c > maxX) maxX = c;
        if (r < minY) minY = r;
        if (r > maxY) maxY = r;
      }
    }
  }
  connected.delete();

  const inkCleanMat = new cv.Mat(rows, cols, cv.CV_8UC1);
  inkCleanMat.data.set(inkClean);

  let cropped: any;
  if (maxX >= minX && maxY >= minY) {
    const pad = 6;
    const x1 = Math.max(0, minX - pad);
    const y1 = Math.max(0, minY - pad);
    const x2 = Math.min(cols, maxX + 1 + pad);
    const y2 = Math.min(rows, maxY + 1 + pad);
    const roi = inkCleanMat.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1));
    cropped = new cv.Mat();
    roi.copyTo(cropped);
    roi.delete();
    inkCleanMat.delete();
  } else {
    cropped = inkCleanMat;
  }

  // 5. Multi-line strip stacking for wide single-line sentences
  const hc = cropped.rows;
  const wc = cropped.cols;
  const aspectRatio = wc / Math.max(hc, 1);

  let nSplits = 1;
  if (aspectRatio > 4.2) nSplits = 3;
  else if (aspectRatio > 2.2) nSplits = 2;

  let block: any;
  if (nSplits > 1 && wc >= nSplits * 20) {
    const segW = Math.floor(wc / nSplits);
    const spacerH = Math.max(6, Math.floor(hc / 8));
    const totalH = nSplits * hc + (nSplits - 1) * spacerH;

    block = new cv.Mat.zeros(totalH, segW, cv.CV_8UC1);
    for (let s = 0; s < nSplits; s++) {
      const srcRoi = cropped.roi(new cv.Rect(s * segW, 0, segW, hc));
      const dstY = s * (hc + spacerH);
      const dstRoi = block.roi(new cv.Rect(0, dstY, segW, hc));
      srcRoi.copyTo(dstRoi);
      srcRoi.delete();
      dstRoi.delete();
    }
    cropped.delete();
  } else {
    block = cropped;
  }

  // 6. Proportional resize onto 224x224 black canvas
  const hb = block.rows;
  const wb = block.cols;
  const usableSize = targetSize - 12;
  const scale = Math.min(usableSize / Math.max(wb, 1), usableSize / Math.max(hb, 1));
  const newW = Math.max(1, Math.floor(wb * scale));
  const newH = Math.max(1, Math.floor(hb * scale));

  const interp = scale > 1.0 ? cv.INTER_CUBIC : cv.INTER_AREA;
  const resized = new cv.Mat();
  cv.resize(block, resized, new cv.Size(newW, newH), 0, 0, interp);
  block.delete();

  const resTotal = newW * newH;
  let rMax = 0;
  let rMin = 255;
  for (let i = 0; i < resTotal; i++) {
    const v = resized.data[i];
    if (v > rMax) rMax = v;
    if (v < rMin) rMin = v;
  }
  if (rMax > 0) {
    const rRange = Math.max(1, rMax - rMin);
    for (let i = 0; i < resTotal; i++) {
      resized.data[i] = Math.round(((resized.data[i] - rMin) * 255.0) / rRange);
    }
  }

  const outCanvas = document.createElement("canvas");
  outCanvas.width = targetSize;
  outCanvas.height = targetSize;
  const outCtx = outCanvas.getContext("2d")!;
  const outImgData = outCtx.createImageData(targetSize, targetSize);

  for (let i = 0; i < targetSize * targetSize; i++) {
    outImgData.data[i * 4 + 3] = 255;
  }

  const yOff = Math.floor((targetSize - newH) / 2);
  const xOff = Math.floor((targetSize - newW) / 2);

  for (let r = 0; r < newH; r++) {
    for (let c = 0; c < newW; c++) {
      const val = resized.data[r * newW + c];
      const dstIdx = ((yOff + r) * targetSize + (xOff + c)) * 4;
      outImgData.data[dstIdx] = val;
      outImgData.data[dstIdx + 1] = val;
      outImgData.data[dstIdx + 2] = val;
      outImgData.data[dstIdx + 3] = 255;
    }
  }

  resized.delete();
  outCtx.putImageData(outImgData, 0, 0);
  return outCanvas;
}

async function getModelProbs(
  model: tf.GraphModel,
  canvas224: HTMLCanvasElement,
): Promise<Float32Array> {
  const channels = model.inputs[0].shape?.[3] ?? 3;

  const input = tf.tidy(() =>
    tf.browser
      .fromPixels(canvas224, channels)
      .resizeBilinear([224, 224])
      .expandDims(0)
      .toFloat(),
  );

  try {
    const prediction = model.predict(input);
    let output: tf.Tensor;

    if (prediction instanceof tf.Tensor) {
      output = prediction;
    } else if (Array.isArray(prediction)) {
      output = prediction[0];
      prediction.slice(1).forEach((t) => t.dispose());
    } else {
      const keys = Object.keys(prediction);
      output = prediction[keys[0]];
      keys.slice(1).forEach((k) => prediction[k].dispose());
    }

    const data = await output.data();
    output.dispose();
    return new Float32Array(data);
  } finally {
    input.dispose();
  }
}

function rankTop5(probs: Float32Array | number[], classNames: string[]): PredictionResult[] {
  return Array.from(probs)
    .map((score, index) => ({
      classIndex: index,
      className: classNames[index] ?? `Class ${index}`,
      score,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
}

export default function InkScanner({ classNames }: InkScannerProps) {
  const [models, setModels] = useState<V2Models>({});
  const [modelStatusText, setModelStatusText] = useState<string>("Loading models...");
  const [cvReady, setCvReady] = useState(false);
  const cvRef = useRef<any>(null);

  const [captured, setCaptured] = useState<Blob | null>(null);
  const [preview, setPreview] = useState("");
  const [showCropper, setShowCropper] = useState(false);

  const [trimBorder, setTrimBorder] = useState(false);
  const [runnerMode, setRunnerMode] = useState<RunnerMode>("local");
  const [serverUrl, setServerUrl] = useState("");
  const [selectedModel, setSelectedModel] = useState<ModelOption>("ensemble");

  const [prepPreviewUrl, setPrepPreviewUrl] = useState("");
  const [predicting, setPredicting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [ensembleTop5, setEnsembleTop5] = useState<PredictionResult[] | null>(null);
  const [effTop5, setEffTop5] = useState<PredictionResult[] | null>(null);
  const [cnnTop5, setCnnTop5] = useState<PredictionResult[] | null>(null);
  const [summaryInfo, setSummaryInfo] = useState<{
    bestClass: string;
    bestScore: number;
    effScore: number;
    cnnScore: number;
  } | null>(null);

  useEffect(() => {
    let alive = true;
    initOpenCv()
      .then(({ cv }) => {
        if (!alive) return;
        cvRef.current = cv;
        setCvReady(true);
      })
      .catch((err) => console.error("OpenCV initialization error:", err));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    const loadedModels: tf.GraphModel[] = [];

    async function loadV2Models() {
      await tf.ready();
      let usingV2Weights = true;

      await Promise.all(
        V2_MODEL_CONFIGS.map(async ({ key, primaryUrl, fallbackUrl }) => {
          try {
            const m = await tf.loadGraphModel(primaryUrl);
            if (!mounted) {
              m.dispose();
              return;
            }
            loadedModels.push(m);
            setModels((prev) => ({ ...prev, [key]: m }));
          } catch {
            usingV2Weights = false;
            try {
              const mFallback = await tf.loadGraphModel(fallbackUrl);
              if (!mounted) {
                mFallback.dispose();
                return;
              }
              loadedModels.push(mFallback);
              setModels((prev) => ({ ...prev, [key]: mFallback }));
            } catch (err2) {
              console.error(`Failed to load ${key}:`, err2);
            }
          }
        }),
      );

      if (mounted) {
        setModelStatusText(
          usingV2Weights
            ? "Ink-Centric Multi-Line Models Ready"
            : "Using default web weights (Place cnn_model_v2 & efficient_model_v2 in public/ for V2 weights)",
        );
      }
    }

    loadV2Models();
    return () => {
      mounted = false;
      loadedModels.forEach((m) => m.dispose());
    };
  }, []);

  useEffect(() => {
    if (!captured) {
    // eslint-disable-next-line react-hooks/set-state-in-effect
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(captured);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [captured]);

  function handleCapture(blob: Blob) {
    setCaptured(blob);
    setShowCropper(false);
    setPrepPreviewUrl("");
    setEnsembleTop5(null);
    setEffTop5(null);
    setCnnTop5(null);
    setSummaryInfo(null);
    setErrorMsg(null);
  }

  async function handleAnalyze() {
    if (!captured || classNames.length !== 49 || predicting) return;

    setPredicting(true);
    setErrorMsg(null);

    try {
      if (runnerMode === "local") {
        if (!cvRef.current) throw new Error("Image processing engine is still loading.");
        if (!models.efficient || !models.cnn) {
          throw new Error("AI models are not ready yet.");
        }

        const bitmap = await createImageBitmap(captured);
        const srcCanvas = document.createElement("canvas");
        srcCanvas.width = bitmap.width;
        srcCanvas.height = bitmap.height;
        srcCanvas.getContext("2d")!.drawImage(bitmap, 0, 0);
        bitmap.close();

        const cleanCanvas224 = buildInkCanvas(
          cvRef.current,
          srcCanvas,
          224,
          trimBorder,
        );
        setPrepPreviewUrl(cleanCanvas224.toDataURL("image/jpeg", 0.95));

        const probsEff = await getModelProbs(models.efficient, cleanCanvas224);
        const probsCnn = await getModelProbs(models.cnn, cleanCanvas224);

        const probsEns = new Float32Array(classNames.length);
        for (let i = 0; i < classNames.length; i++) {
          probsEns[i] = 0.6 * probsEff[i] + 0.4 * probsCnn[i];
        }

        const top5Ens = rankTop5(probsEns, classNames);
        const top5Eff = rankTop5(probsEff, classNames);
        const top5Cnn = rankTop5(probsCnn, classNames);

        const activeTop =
          selectedModel === "ensemble"
            ? top5Ens[0]
            : selectedModel === "efficient"
              ? top5Eff[0]
              : top5Cnn[0];

        setEnsembleTop5(top5Ens);
        setEffTop5(top5Eff);
        setCnnTop5(top5Cnn);
        setSummaryInfo({
          bestClass: activeTop.className,
          bestScore: activeTop.score * 100,
          effScore: probsEff[activeTop.classIndex] * 100,
          cnnScore: probsCnn[activeTop.classIndex] * 100,
        });
      } else {
        const baseUrl = serverUrl.trim().replace(/\/$/, "");
        if (!baseUrl.startsWith("http")) {
          throw new Error("Please enter a valid server URL.");
        }

        const formData = new FormData();
        formData.append("files", captured, "handwriting.jpg");

        const uploadRes = await fetch(`${baseUrl}/gradio_api/upload`, {
          method: "POST",
          body: formData,
        });
        if (!uploadRes.ok) {
          throw new Error("Failed to upload image to remote server.");
        }
        const uploadedPaths: string[] = await uploadRes.json();
        const filePath = uploadedPaths[0];

        const modeLabel =
          selectedModel === "ensemble"
            ? "Ensemble Terbaik (EfficientNetB0 + Custom CNN)"
            : selectedModel === "efficient"
              ? "EfficientNetB0 (Fine-Tuned - Val Acc 94.89%)"
              : "Custom CNN + Dense Network (Val Acc 94.51%)";

        const callRes = await fetch(`${baseUrl}/gradio_api/call/klasifikasi_tulisan_tangan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            data: [
              { path: filePath, meta: { _type: "gradio.FileData" } },
              modeLabel,
              trimBorder,
            ],
          }),
        });
        if (!callRes.ok) {
          throw new Error("Prediction request failed on remote server.");
        }
        const { event_id } = await callRes.json();

        const streamRes = await fetch(
          `${baseUrl}/gradio_api/call/klasifikasi_tulisan_tangan/${event_id}`,
        );
        const streamText = await streamRes.text();
        const dataLine = streamText
          .split("\n")
          .reverse()
          .find((line) => line.startsWith("data: "));

        if (!dataLine) {
          throw new Error("Empty response from remote server.");
        }

        const parsedData = JSON.parse(dataLine.slice(6));
        const [prepImgObj, labelObj] = parsedData;

        if (prepImgObj?.url) {
          setPrepPreviewUrl(prepImgObj.url);
        } else if (prepImgObj?.path) {
          setPrepPreviewUrl(`${baseUrl}/gradio_api/file=${prepImgObj.path}`);
        }

        const confidences: Array<{ label: string; confidence: number }> =
          labelObj?.confidences ?? [];
        const mappedTop5: PredictionResult[] = confidences.map((item, idx) => ({
          classIndex: idx,
          className: item.label,
          score: item.confidence,
        }));

        setEnsembleTop5(mappedTop5);
        setEffTop5(mappedTop5);
        setCnnTop5(mappedTop5);

        if (mappedTop5.length > 0) {
          setSummaryInfo({
            bestClass: mappedTop5[0].className,
            bestScore: mappedTop5[0].score * 100,
            effScore: mappedTop5[0].score * 100,
            cnnScore: mappedTop5[0].score * 100,
          });
        }
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Analysis failed.");
    } finally {
      setPredicting(false);
    }
  }

  const displayedResults =
    selectedModel === "ensemble"
      ? ensembleTop5
      : selectedModel === "efficient"
        ? effTop5
        : cnnTop5;

  return (
    <div className='wizard-card'>
      <div className='card-heading'>
        <div>
          <span className='step-caption'>INK-CENTRIC MULTI-LINE PIPELINE</span>
          <h2>Enhanced Handwriting Scanner</h2>
          <p>{modelStatusText}</p>
        </div>
        <div className='step-number'>V2</div>
      </div>

      <div className='wizard-content'>
        {/* Execution mode bar */}
        <div className='runner-bar'>
          <div className='runner-tabs'>
            <button
              type='button'
              className={runnerMode === "local" ? "active" : ""}
              onClick={() => setRunnerMode("local")}
            >
              Browser Engine (TensorFlow.js + OpenCV.js)
            </button>
            <button
              type='button'
              className={runnerMode === "remote" ? "active" : ""}
              onClick={() => setRunnerMode("remote")}
            >
              Remote Python Server (API)
            </button>
          </div>

          {runnerMode === "remote" && (
            <div className='remote-url-input'>
              <label>Server Endpoint URL:</label>
              <input
                type='url'
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                placeholder='https://xxxx.gradio.live'
              />
            </div>
          )}
        </div>

        <div className='content-intro'>
          <h3>Capture or upload handwriting</h3>
          <p>
            Automatically removes paper shadows, inverts ink polarity, and stacks
            long sentences into a 224 × 224 canvas.
          </p>
        </div>

        <div className='camera-container'>
          <Camera onCapture={handleCapture} />
        </div>

        {preview && (
          <div className='image-preview-card'>
            <div className='preview-heading'>
              <strong>Input image</strong>
              <span className='success-label'>Ready</span>
            </div>
            <img src={preview} alt='Captured handwriting' />
            <button
              type='button'
              className='secondary-button'
              onClick={() => setShowCropper(true)}
            >
              Crop image
            </button>
          </div>
        )}

        {showCropper && preview && (
          <ImageCropper
            image={preview}
            onCrop={handleCapture}
            onCancel={() => setShowCropper(false)}
          />
        )}

        {preview && (
          <div className='scanner-options-card'>
            <h4>Model & Preprocessing Options</h4>

            <div className='model-option-grid'>
              {[
                {
                  id: "ensemble",
                  title: "Ensemble (EfficientNetB0 + CNN)",
                },
                {
                  id: "efficient",
                  title: "EfficientNetB0 (Fine-Tuned)",
                },
                {
                  id: "cnn",
                  title: "Custom CNN + Dense Network",
                },
              ].map((m) => (
                <button
                  key={m.id}
                  type='button'
                  className={`model-option-btn ${selectedModel === m.id ? "active" : ""}`}
                  onClick={() => setSelectedModel(m.id as ModelOption)}
                >
                  {m.title}
                </button>
              ))}
            </div>

            <label className='trim-checkbox'>
              <input
                type='checkbox'
                checked={trimBorder}
                onChange={(e) => setTrimBorder(e.target.checked)}
              />
              <span>
                Auto-trim dark background border (recommended for mobile camera photos)
              </span>
            </label>

            <button
              type='button'
              className='primary-button analyze-button'
              onClick={handleAnalyze}
              disabled={predicting || (!cvReady && runnerMode === "local")}
            >
              {predicting ? "Analyzing image..." : "✦ Analyze Handwriting"}
            </button>
          </div>
        )}

        {errorMsg && <p className='error' style={{ marginTop: 16 }}>{errorMsg}</p>}

        {summaryInfo && displayedResults && (
          <div className='results-container'>
            <div className='summary-banner'>
              <div>
                <span className='eyebrow'>TOP PREDICTION</span>
                <h3>{summaryInfo.bestClass}</h3>
                <p>
                  Confidence: <strong>{summaryInfo.bestScore.toFixed(2)}%</strong>{" "}
                  {runnerMode === "local" && (
                    <span>
                      (EfficientNetB0: {summaryInfo.effScore.toFixed(2)}% | CNN:{" "}
                      {summaryInfo.cnnScore.toFixed(2)}%)
                    </span>
                  )}
                </p>
              </div>

              {prepPreviewUrl && (
                <div className='prep-preview-box'>
                  <span>Model Input (224×224)</span>
                  <img src={prepPreviewUrl} alt='Preprocessed input' />
                </div>
              )}
            </div>

            {runnerMode === "local" && ensembleTop5 && effTop5 && cnnTop5 ? (
              <div className='results-three-cols'>
                {[
                  { title: "Ensemble", data: ensembleTop5 },
                  { title: "EfficientNetB0", data: effTop5 },
                  { title: "CNN", data: cnnTop5 },
                ].map((col) => (
                  <div className='model-results' key={col.title}>
                    <h4 className='model-results-title'>{col.title}</h4>
                    {col.data.map((pred, idx) => (
                      <div
                        className={`prediction-item ${idx === 0 ? "top-prediction" : ""}`}
                        key={pred.classIndex}
                      >
                        <div className='prediction-rank'>{idx + 1}</div>
                        <div className='prediction-details'>
                          <div className='prediction-name'>
                            <strong>{pred.className}</strong>
                            <span>{(pred.score * 100).toFixed(2)}%</span>
                          </div>
                          <div className='prediction-bar'>
                            <div
                              className='prediction-fill'
                              style={{
                                width: `${Math.min(100, Math.max(0, pred.score * 100))}%`,
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            ) : (
              <div className='model-results'>
                <h4 className='model-results-title'>Top 5 Predictions</h4>
                {displayedResults.map((pred, idx) => (
                  <div
                    className={`prediction-item ${idx === 0 ? "top-prediction" : ""}`}
                    key={pred.classIndex}
                  >
                    <div className='prediction-rank'>{idx + 1}</div>
                    <div className='prediction-details'>
                      <div className='prediction-name'>
                        <strong>{pred.className}</strong>
                        <span>{(pred.score * 100).toFixed(2)}%</span>
                      </div>
                      <div className='prediction-bar'>
                        <div
                          className='prediction-fill'
                          style={{
                            width: `${Math.min(100, Math.max(0, pred.score * 100))}%`,
                          }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}