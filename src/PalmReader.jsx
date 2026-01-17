import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  analyzeWithVLM,
  generateVLMGuidedMask,
  generatePalmReading,
  LINE_COLORS,
  getLineColor,
} from './vlmService';

// Skin detection using YCbCr color space - works well for various skin tones
const createSkinMask = (data, width, height) => {
  const mask = new Uint8Array(width * height);

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const idx = i / 4;

    // Convert RGB to YCbCr
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
    const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;

    // Skin detection thresholds in YCbCr space (works for various skin tones)
    const isSkin = (
      y > 80 &&
      cb > 77 && cb < 127 &&
      cr > 133 && cr < 173
    );

    mask[idx] = isSkin ? 255 : 0;
  }

  return mask;
};

// Morphological erosion to shrink mask
const erodeMask = (mask, width, height, radius) => {
  const result = new Uint8Array(width * height);

  for (let y = radius; y < height - radius; y++) {
    for (let x = radius; x < width - radius; x++) {
      const idx = y * width + x;
      let allSet = true;

      // Check if all pixels in the radius are set
      outer: for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy <= radius * radius) {
            if (mask[(y + dy) * width + (x + dx)] === 0) {
              allSet = false;
              break outer;
            }
          }
        }
      }

      result[idx] = allSet ? 255 : 0;
    }
  }

  return result;
};

// Create palm region mask - very restrictive to only include inner palm
const createPalmMask = (data, width, height) => {
  // Step 1: Detect skin pixels
  const skinMask = createSkinMask(data, width, height);

  // Step 2: Erode VERY aggressively to get only the innermost palm region
  // This removes fingers completely and leaves only the palm center
  const erodeRadius = Math.max(15, Math.floor(Math.min(width, height) / 12));
  const erodedMask = erodeMask(skinMask, width, height, erodeRadius);

  // Step 3: Find bounding box of skin region
  let skinMinY = height, skinMaxY = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (skinMask[y * width + x] === 255) {
        skinMinY = Math.min(skinMinY, y);
        skinMaxY = Math.max(skinMaxY, y);
      }
    }
  }

  // Step 4: Find the palm center from eroded mask
  let sumX = 0, sumY = 0, count = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (erodedMask[y * width + x] === 255) {
        sumX += x;
        sumY += y;
        count++;
      }
    }
  }

  // If no palm detected after erosion, use fallback
  if (count < 50) {
    const fallbackMask = new Uint8Array(width * height);
    const centerX = width / 2;
    const centerY = height * 0.55; // Lower center for palm
    const radiusX = width * 0.25;
    const radiusY = height * 0.25;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const dx = (x - centerX) / radiusX;
        const dy = (y - centerY) / radiusY;
        if (dx * dx + dy * dy <= 1) {
          fallbackMask[y * width + x] = 255;
        }
      }
    }
    return fallbackMask;
  }

  const palmCenterX = sumX / count;
  const palmCenterY = sumY / count;

  // Step 5: Create final palm mask - elliptical region around palm center
  // Exclude upper portion (fingers) by limiting vertical extent
  const palmMask = new Uint8Array(width * height);
  const handHeight = skinMaxY - skinMinY;
  const palmRadiusX = width * 0.3;
  const palmRadiusY = handHeight * 0.35;

  // The palm region should be in the lower-middle of the hand
  // Shift center down slightly to avoid fingers
  const adjustedCenterY = Math.min(palmCenterY + handHeight * 0.05, skinMaxY - palmRadiusY);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;

      // Elliptical palm region
      const dx = (x - palmCenterX) / palmRadiusX;
      const dy = (y - adjustedCenterY) / palmRadiusY;

      if (dx * dx + dy * dy <= 1 && skinMask[idx] === 255) {
        palmMask[idx] = 255;
      }
    }
  }

  return palmMask;
};

// Palm line detection - optimized for major palm creases only
const detectPalmLines = (imageData, sensitivity = 50, lineThickness = 2, vlmMask = null, vlmWeight = 0.5) => {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  const output = new Uint8ClampedArray(data.length);

  // Create restrictive palm region mask
  const palmMask = createPalmMask(data, width, height);

  // Convert to grayscale
  for (let i = 0; i < data.length; i += 4) {
    const idx = i / 4;
    gray[idx] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  // Apply contrast enhancement
  const contrastFactor = 1.2;
  for (let i = 0; i < gray.length; i++) {
    gray[i] = Math.max(0, Math.min(255, ((gray[i] - 128) * contrastFactor) + 128));
  }

  // Apply strong Gaussian blur to eliminate skin texture
  const kernel = [1, 4, 6, 4, 1];
  const kernelSum = 16;

  const applyGaussianBlur = (input) => {
    const temp = new Float32Array(width * height);
    const result = new Float32Array(width * height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        for (let k = -2; k <= 2; k++) {
          const px = Math.min(Math.max(x + k, 0), width - 1);
          sum += input[y * width + px] * kernel[k + 2];
        }
        temp[y * width + x] = sum / kernelSum;
      }
    }

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        for (let k = -2; k <= 2; k++) {
          const py = Math.min(Math.max(y + k, 0), height - 1);
          sum += temp[py * width + x] * kernel[k + 2];
        }
        result[y * width + x] = sum / kernelSum;
      }
    }

    return result;
  };

  // Apply blur 3 times for very strong smoothing
  let blurred = applyGaussianBlur(gray);
  blurred = applyGaussianBlur(blurred);
  blurred = applyGaussianBlur(blurred);

  // Detect valleys (dark creases) - palm lines are darker than surrounding skin
  const valleys = new Float32Array(width * height);

  for (let y = 3; y < height - 3; y++) {
    for (let x = 3; x < width - 3; x++) {
      const idx = y * width + x;
      if (palmMask[idx] === 0) continue;

      const center = blurred[idx];

      // Sample neighbors at larger distance for better crease detection
      const neighbors = [
        blurred[(y - 3) * width + x],
        blurred[(y + 3) * width + x],
        blurred[y * width + (x - 3)],
        blurred[y * width + (x + 3)],
        blurred[(y - 3) * width + (x - 3)],
        blurred[(y - 3) * width + (x + 3)],
        blurred[(y + 3) * width + (x - 3)],
        blurred[(y + 3) * width + (x + 3)],
      ];

      // Valley score - how much darker is center vs neighbors
      let valleyScore = 0;
      for (const neighbor of neighbors) {
        if (neighbor > center) {
          valleyScore += neighbor - center;
        }
      }

      valleys[idx] = valleyScore;
    }
  }

  // Sobel edge detection
  const edges = new Float32Array(width * height);
  const directions = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      if (palmMask[idx] === 0) continue;

      const gx = (
        -blurred[(y - 1) * width + (x - 1)] - 2 * blurred[y * width + (x - 1)] - blurred[(y + 1) * width + (x - 1)] +
        blurred[(y - 1) * width + (x + 1)] + 2 * blurred[y * width + (x + 1)] + blurred[(y + 1) * width + (x + 1)]
      );

      const gy = (
        -blurred[(y - 1) * width + (x - 1)] - 2 * blurred[(y - 1) * width + x] - blurred[(y - 1) * width + (x + 1)] +
        blurred[(y + 1) * width + (x - 1)] + 2 * blurred[(y + 1) * width + x] + blurred[(y + 1) * width + (x + 1)]
      );

      edges[idx] = Math.sqrt(gx * gx + gy * gy);
      directions[idx] = Math.atan2(gy, gx);
    }
  }

  // Non-maximum suppression
  const suppressed = new Float32Array(width * height);

  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      const idx = y * width + x;
      if (palmMask[idx] === 0) continue;

      const angle = directions[idx];
      const mag = edges[idx];

      let neighbor1, neighbor2;
      const sector = Math.round(((angle + Math.PI) / Math.PI) * 4) % 4;

      switch (sector) {
        case 0:
          neighbor1 = edges[y * width + (x - 1)];
          neighbor2 = edges[y * width + (x + 1)];
          break;
        case 1:
          neighbor1 = edges[(y - 1) * width + (x + 1)];
          neighbor2 = edges[(y + 1) * width + (x - 1)];
          break;
        case 2:
          neighbor1 = edges[(y - 1) * width + x];
          neighbor2 = edges[(y + 1) * width + x];
          break;
        case 3:
          neighbor1 = edges[(y - 1) * width + (x - 1)];
          neighbor2 = edges[(y + 1) * width + (x + 1)];
          break;
        default:
          neighbor1 = neighbor2 = 0;
      }

      if (mag >= neighbor1 && mag >= neighbor2) {
        // Strongly boost edges that are valleys (dark creases)
        // Suppress edges that aren't valleys (skin texture, etc.)
        const valleyBoost = valleys[idx] > 5 ? 1.5 + (valleys[idx] / 50) : 0.1;
        suppressed[idx] = mag * valleyBoost;
      }
    }
  }

  // Find max edge for threshold calculation
  let maxEdge = 0;
  for (let i = 0; i < suppressed.length; i++) {
    if (suppressed[i] > maxEdge) maxEdge = suppressed[i];
  }

  // Very high threshold - only detect the strongest creases
  const sensitivityFactor = (sensitivity - 20) / 75;
  const baseThreshold = 0.35; // Much higher base threshold
  const minThreshold = 0.15;  // Higher minimum too
  const thresholdMultiplier = baseThreshold - (sensitivityFactor * (baseThreshold - minThreshold));
  const threshold = maxEdge * thresholdMultiplier;
  const lowThreshold = threshold * 0.6;

  const result = new Uint8Array(width * height);

  // Hysteresis thresholding
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (palmMask[idx] === 0) continue;

      let effectiveThreshold = threshold;
      let effectiveLowThreshold = lowThreshold;

      if (vlmMask && vlmMask[idx] > 0) {
        const vlmFactor = 1 - (vlmWeight * 0.4);
        effectiveThreshold = threshold * vlmFactor;
        effectiveLowThreshold = lowThreshold * vlmFactor;
      }

      if (suppressed[idx] > effectiveThreshold) {
        result[idx] = 255;
      } else if (suppressed[idx] > effectiveLowThreshold) {
        let connected = false;
        for (let dy = -1; dy <= 1 && !connected; dy++) {
          for (let dx = -1; dx <= 1 && !connected; dx++) {
            const ny = y + dy;
            const nx = x + dx;
            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
              if (suppressed[ny * width + nx] > effectiveThreshold) {
                connected = true;
              }
            }
          }
        }
        if (connected) result[idx] = 255;
      }
    }
  }

  // Apply dilation
  const dilated = new Uint8Array(width * height);
  const dilateRadius = Math.max(0, Math.floor((lineThickness - 1) / 2));

  if (dilateRadius === 0) {
    dilated.set(result);
  } else {
    for (let y = dilateRadius; y < height - dilateRadius; y++) {
      for (let x = dilateRadius; x < width - dilateRadius; x++) {
        if (result[y * width + x] === 255) {
          for (let dy = -dilateRadius; dy <= dilateRadius; dy++) {
            for (let dx = -dilateRadius; dx <= dilateRadius; dx++) {
              if (dx * dx + dy * dy <= dilateRadius * dilateRadius + 1) {
                const targetIdx = (y + dy) * width + (x + dx);
                if (palmMask[targetIdx] === 255) {
                  dilated[targetIdx] = 255;
                }
              }
            }
          }
        }
      }
    }
  }

  // Output golden colored lines
  for (let i = 0; i < data.length; i += 4) {
    const idx = i / 4;
    if (dilated[idx] === 255) {
      output[i] = 255;
      output[i + 1] = 200;
      output[i + 2] = 100;
      output[i + 3] = 255;
    } else {
      output[i] = 0;
      output[i + 1] = 0;
      output[i + 2] = 0;
      output[i + 3] = 0;
    }
  }

  return new ImageData(output, width, height);
};

const StarField = () => {
  const stars = Array.from({ length: 50 }, (_, i) => ({
    id: i,
    left: Math.random() * 100,
    top: Math.random() * 100,
    size: Math.random() * 2 + 1,
    delay: Math.random() * 3,
    duration: Math.random() * 2 + 2,
  }));

  return (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none' }}>
      {stars.map(star => (
        <div
          key={star.id}
          style={{
            position: 'absolute',
            left: `${star.left}%`,
            top: `${star.top}%`,
            width: star.size,
            height: star.size,
            background: 'radial-gradient(circle, rgba(255,215,150,0.9) 0%, transparent 70%)',
            borderRadius: '50%',
            animation: `twinkle ${star.duration}s ease-in-out ${star.delay}s infinite`,
          }}
        />
      ))}
    </div>
  );
};

// Component to draw annotated lines on canvas
const LineAnnotationCanvas = ({ vlmResult, width, height }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !vlmResult?.lines?.length) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    vlmResult.lines.forEach(line => {
      if (!line.points || line.points.length < 2) return;

      const color = getLineColor(line.name);
      ctx.strokeStyle = color.hex;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.shadowColor = color.hex;
      ctx.shadowBlur = 8;

      // Convert percentage points to pixel coordinates
      const pixelPoints = line.points.map(p => ({
        x: (p.x / 100) * width,
        y: (p.y / 100) * height,
      }));

      // Draw smooth curve using quadratic bezier
      ctx.beginPath();
      ctx.moveTo(pixelPoints[0].x, pixelPoints[0].y);

      for (let i = 1; i < pixelPoints.length - 1; i++) {
        const xc = (pixelPoints[i].x + pixelPoints[i + 1].x) / 2;
        const yc = (pixelPoints[i].y + pixelPoints[i + 1].y) / 2;
        ctx.quadraticCurveTo(pixelPoints[i].x, pixelPoints[i].y, xc, yc);
      }

      // Draw last segment
      if (pixelPoints.length > 1) {
        const last = pixelPoints[pixelPoints.length - 1];
        ctx.lineTo(last.x, last.y);
      }

      ctx.stroke();

      // Draw label at midpoint
      const midIndex = Math.floor(pixelPoints.length / 2);
      const midPoint = pixelPoints[midIndex];

      ctx.shadowBlur = 0;
      ctx.font = 'bold 11px "Cinzel", serif';
      ctx.fillStyle = color.hex;
      ctx.textAlign = 'center';

      // Background for label
      const labelWidth = ctx.measureText(line.name).width + 10;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(midPoint.x - labelWidth / 2, midPoint.y - 20, labelWidth, 16);

      ctx.fillStyle = color.hex;
      ctx.fillText(line.name, midPoint.x, midPoint.y - 8);
    });
  }, [vlmResult, width, height]);

  if (!vlmResult?.lines?.length) return null;

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
      }}
    />
  );
};

// Reading Section Component
const ReadingSection = ({ reading, isGenerating }) => {
  if (isGenerating) {
    return (
      <div style={{
        padding: '40px 30px',
        textAlign: 'center',
      }}>
        <div style={{
          width: 50,
          height: 50,
          margin: '0 auto 20px',
          border: '3px solid rgba(212, 175, 55, 0.2)',
          borderTopColor: '#d4af37',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
        }} />
        <p style={{
          fontFamily: '"Cinzel", serif',
          fontSize: '1rem',
          letterSpacing: '0.1em',
          color: '#d4af37',
        }}>
          Channeling the wisdom of the lines...
        </p>
      </div>
    );
  }

  if (!reading) return null;

  const renderLineReading = (lineData, icon) => {
    if (!lineData) return null;
    return (
      <div style={{
        marginBottom: 25,
        padding: '20px',
        background: 'rgba(20, 15, 35, 0.6)',
        borderRadius: 12,
        border: '1px solid rgba(212, 175, 55, 0.15)',
      }}>
        <h4 style={{
          fontFamily: '"Cinzel", serif',
          fontSize: '1rem',
          color: '#d4af37',
          marginTop: 0,
          marginBottom: 12,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}>
          <span>{icon}</span>
          {lineData.title || lineData.name}
        </h4>
        <p style={{
          fontSize: '0.95rem',
          lineHeight: 1.8,
          color: 'rgba(232, 220, 200, 0.85)',
          margin: 0,
        }}>
          {lineData.interpretation}
        </p>
      </div>
    );
  };

  return (
    <div style={{
      padding: '30px',
      animation: 'fadeIn 0.8s ease-out',
    }}>
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      {/* Opening */}
      {reading.opening && (
        <div style={{
          textAlign: 'center',
          marginBottom: 30,
          padding: '25px',
          background: 'linear-gradient(135deg, rgba(139, 105, 20, 0.15), rgba(212, 175, 55, 0.1))',
          borderRadius: 15,
          border: '1px solid rgba(212, 175, 55, 0.25)',
        }}>
          <p style={{
            fontFamily: '"Cormorant Garamond", serif',
            fontSize: '1.15rem',
            fontStyle: 'italic',
            lineHeight: 1.8,
            color: '#f5d76e',
            margin: 0,
          }}>
            "{reading.opening}"
          </p>
        </div>
      )}

      {/* Main Line Readings */}
      {renderLineReading(reading.heartLine, '❤')}
      {renderLineReading(reading.headLine, '🧠')}
      {renderLineReading(reading.lifeLine, '✨')}

      {/* Other Lines */}
      {reading.otherLines && reading.otherLines.length > 0 && (
        <div style={{ marginBottom: 25 }}>
          {reading.otherLines.map((line, idx) => (
            <div key={idx} style={{
              marginBottom: 15,
              padding: '15px 20px',
              background: 'rgba(20, 15, 35, 0.5)',
              borderRadius: 10,
              border: '1px solid rgba(212, 175, 55, 0.1)',
            }}>
              <h5 style={{
                fontFamily: '"Cinzel", serif',
                fontSize: '0.9rem',
                color: '#d4af37',
                marginTop: 0,
                marginBottom: 8,
              }}>
                {line.title || line.name}
              </h5>
              <p style={{
                fontSize: '0.9rem',
                lineHeight: 1.7,
                color: 'rgba(232, 220, 200, 0.8)',
                margin: 0,
              }}>
                {line.interpretation}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Special Features */}
      {reading.specialFeatures && (
        <div style={{
          marginBottom: 25,
          padding: '15px 20px',
          background: 'rgba(100, 80, 150, 0.15)',
          borderRadius: 10,
          border: '1px solid rgba(150, 120, 200, 0.2)',
        }}>
          <h5 style={{
            fontFamily: '"Cinzel", serif',
            fontSize: '0.9rem',
            color: '#c0a0ff',
            marginTop: 0,
            marginBottom: 8,
          }}>
            Special Markings
          </h5>
          <p style={{
            fontSize: '0.9rem',
            lineHeight: 1.7,
            color: 'rgba(232, 220, 200, 0.8)',
            margin: 0,
          }}>
            {reading.specialFeatures}
          </p>
        </div>
      )}

      {/* Overall Reading */}
      {reading.overallReading && (
        <div style={{
          marginBottom: 25,
          padding: '25px',
          background: 'linear-gradient(135deg, rgba(20, 15, 35, 0.8), rgba(30, 25, 50, 0.6))',
          borderRadius: 15,
          border: '1px solid rgba(212, 175, 55, 0.2)',
        }}>
          <h4 style={{
            fontFamily: '"Cinzel", serif',
            fontSize: '1.1rem',
            color: '#f5d76e',
            marginTop: 0,
            marginBottom: 15,
            textAlign: 'center',
          }}>
            ✧ The Complete Picture ✧
          </h4>
          <p style={{
            fontSize: '1rem',
            lineHeight: 1.9,
            color: 'rgba(232, 220, 200, 0.9)',
            margin: 0,
            textAlign: 'center',
          }}>
            {reading.overallReading}
          </p>
        </div>
      )}

      {/* Guidance */}
      {reading.guidance && (
        <div style={{
          padding: '25px',
          background: 'linear-gradient(135deg, rgba(139, 105, 20, 0.2), rgba(212, 175, 55, 0.1))',
          borderRadius: 15,
          border: '2px solid rgba(212, 175, 55, 0.3)',
          textAlign: 'center',
        }}>
          <h4 style={{
            fontFamily: '"Cinzel", serif',
            fontSize: '1rem',
            color: '#d4af37',
            marginTop: 0,
            marginBottom: 12,
          }}>
            Guidance for Your Path
          </h4>
          <p style={{
            fontFamily: '"Cormorant Garamond", serif',
            fontSize: '1.1rem',
            fontStyle: 'italic',
            lineHeight: 1.8,
            color: '#f5d76e',
            margin: 0,
          }}>
            {reading.guidance}
          </p>
        </div>
      )}
    </div>
  );
};

export default function PalmReader() {
  const [image, setImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [sensitivity, setSensitivity] = useState(60);
  const [lineThickness, setLineThickness] = useState(2);
  const [showOriginal, setShowOriginal] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [displayedDimensions, setDisplayedDimensions] = useState({ width: 0, height: 0 });
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const displayedImageRef = useRef(null);

  // VLM-related state
  const [vlmEnabled, setVlmEnabled] = useState(false);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('anthropic_api_key') || '');
  const [vlmResult, setVlmResult] = useState(null);
  const [vlmMask, setVlmMask] = useState(null);
  const [vlmWeight, setVlmWeight] = useState(0.6);
  const [vlmProcessing, setVlmProcessing] = useState(false);
  const [vlmError, setVlmError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showLineAnnotations, setShowLineAnnotations] = useState(false);

  // Reading state
  const [palmReading, setPalmReading] = useState(null);
  const [isGeneratingReading, setIsGeneratingReading] = useState(false);
  const [showReading, setShowReading] = useState(false);

  // Animation state for replay
  const [animationKey, setAnimationKey] = useState(0);

  // File validation constants
  const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
  const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

  useEffect(() => {
    if (apiKey) {
      localStorage.setItem('anthropic_api_key', apiKey);
    }
  }, [apiKey]);

  // Update displayed dimensions when image renders
  const updateDisplayedDimensions = useCallback(() => {
    if (displayedImageRef.current) {
      const { clientWidth, clientHeight } = displayedImageRef.current;
      if (clientWidth > 0 && clientHeight > 0) {
        setDisplayedDimensions({ width: clientWidth, height: clientHeight });
      }
    }
  }, []);

  // Update dimensions on window resize
  useEffect(() => {
    window.addEventListener('resize', updateDisplayedDimensions);
    return () => window.removeEventListener('resize', updateDisplayedDimensions);
  }, [updateDisplayedDimensions]);

  const processImage = useCallback((imgSrc, sens = sensitivity, thickness = lineThickness, currentVlmMask = vlmMask, currentVlmWeight = vlmWeight) => {
    setIsProcessing(true);

    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      const maxSize = 800;
      let { width, height } = img;
      if (width > maxSize || height > maxSize) {
        const ratio = Math.min(maxSize / width, maxSize / height);
        width = Math.round(width * ratio);
        height = Math.round(height * ratio);
      }

      setImageDimensions({ width, height });

      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0, width, height);

      const imageData = ctx.getImageData(0, 0, width, height);
      const maskToUse = vlmEnabled && currentVlmMask ? currentVlmMask : null;
      const processed = detectPalmLines(imageData, sens, thickness, maskToUse, currentVlmWeight);

      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = width;
      tempCanvas.height = height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.putImageData(processed, 0, 0);

      setProcessedImage(tempCanvas.toDataURL());
      setIsProcessing(false);
    };
    img.src = imgSrc;
  }, [sensitivity, lineThickness, vlmMask, vlmWeight, vlmEnabled]);

  const runVLMAnalysis = useCallback(async (imgSrc) => {
    if (!apiKey) {
      setVlmError('Please enter an API key in settings');
      setShowSettings(true);
      return;
    }

    setVlmProcessing(true);
    setVlmError(null);

    try {
      const result = await analyzeWithVLM(imgSrc, apiKey);
      setVlmResult(result);

      if (result.lines && result.lines.length > 0) {
        const img = new Image();
        img.onload = () => {
          const maxSize = 800;
          let { width, height } = img;
          if (width > maxSize || height > maxSize) {
            const ratio = Math.min(maxSize / width, maxSize / height);
            width = Math.round(width * ratio);
            height = Math.round(height * ratio);
          }

          const mask = generateVLMGuidedMask(result, width, height, 12);
          setVlmMask(mask);
          processImage(imgSrc, sensitivity, lineThickness, mask, vlmWeight);
        };
        img.src = imgSrc;
      }

      setVlmProcessing(false);
    } catch (err) {
      console.error('VLM analysis error:', err);
      setVlmError(err.message);
      setVlmProcessing(false);
    }
  }, [apiKey, sensitivity, lineThickness, vlmWeight, processImage]);

  const generateReading = useCallback(async () => {
    if (!vlmResult || !apiKey) {
      setVlmError('Please analyze the palm first with AI Vision');
      return;
    }

    setIsGeneratingReading(true);
    setShowReading(true);

    try {
      const reading = await generatePalmReading(vlmResult, apiKey);
      setPalmReading(reading);
    } catch (err) {
      console.error('Reading generation error:', err);
      setVlmError(err.message);
    }

    setIsGeneratingReading(false);
  }, [vlmResult, apiKey]);

  // File validation function
  const validateFile = useCallback((file) => {
    if (!file) {
      return { valid: false, error: 'No file selected' };
    }
    if (!ALLOWED_MIME_TYPES.includes(file.type)) {
      return { valid: false, error: 'Invalid file type. Please upload a JPEG, PNG, or WebP image.' };
    }
    if (file.size > MAX_FILE_SIZE) {
      return { valid: false, error: 'File too large. Maximum size is 10MB.' };
    }
    return { valid: true, error: null };
  }, []);

  // Download result function
  const downloadResult = useCallback(() => {
    if (!image || !processedImage) return;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const img = new Image();
    img.onload = () => {
      canvas.width = imageDimensions.width;
      canvas.height = imageDimensions.height;

      // Draw original image
      ctx.drawImage(img, 0, 0, imageDimensions.width, imageDimensions.height);

      // Draw processed overlay
      const overlayImg = new Image();
      overlayImg.onload = () => {
        ctx.globalCompositeOperation = 'screen';
        ctx.drawImage(overlayImg, 0, 0);

        // Create download link
        const link = document.createElement('a');
        link.download = 'palm-reading-result.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
      };
      overlayImg.src = processedImage;
    };
    img.src = image;
  }, [image, processedImage, imageDimensions]);

  // Replay animation function
  const replayAnimation = useCallback(() => {
    setAnimationKey(prev => prev + 1);
  }, []);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const validation = validateFile(file);
      if (!validation.valid) {
        setVlmError(validation.error);
        return;
      }
      setVlmError(null);
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        setVlmResult(null);
        setVlmMask(null);
        setPalmReading(null);
        setShowReading(false);
        setAnimationKey(prev => prev + 1);
        processImage(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      const validation = validateFile(file);
      if (!validation.valid) {
        setVlmError(validation.error);
        return;
      }
      setVlmError(null);
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        setVlmResult(null);
        setVlmMask(null);
        setPalmReading(null);
        setShowReading(false);
        setAnimationKey(prev => prev + 1);
        processImage(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  useEffect(() => {
    if (image) {
      processImage(image, sensitivity, lineThickness, vlmMask, vlmWeight);
    }
  }, [sensitivity, lineThickness, vlmWeight, vlmEnabled]);

  useEffect(() => {
    if (vlmEnabled && image && !vlmResult && apiKey) {
      runVLMAnalysis(image);
    }
  }, [vlmEnabled, image, apiKey]);

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(160deg, #0a0a12 0%, #1a1025 30%, #0d1520 70%, #0a0a12 100%)',
      fontFamily: '"Cormorant Garamond", Georgia, serif',
      color: '#e8dcc8',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&family=Cinzel:wght@400;500;600&display=swap');

        @keyframes twinkle {
          0%, 100% { opacity: 0.3; transform: scale(1); }
          50% { opacity: 1; transform: scale(1.2); }
        }

        @keyframes float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          50% { transform: translateY(-10px) rotate(2deg); }
        }

        @keyframes pulse {
          0%, 100% { opacity: 0.6; }
          50% { opacity: 1; }
        }

        @keyframes shimmer {
          0% { background-position: -200% center; }
          100% { background-position: 200% center; }
        }

        @keyframes revealLines {
          0% { clip-path: circle(0% at 50% 50%); }
          100% { clip-path: circle(100% at 50% 50%); }
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .mystical-border {
          position: relative;
        }

        .mystical-border::before {
          content: '';
          position: absolute;
          inset: -2px;
          background: linear-gradient(45deg, #8b6914, #d4af37, #f5d76e, #d4af37, #8b6914);
          background-size: 300% 300%;
          animation: shimmer 4s linear infinite;
          border-radius: inherit;
          z-index: -1;
          opacity: 0.7;
        }

        .upload-zone:hover {
          border-color: #d4af37 !important;
          background: rgba(212, 175, 55, 0.05) !important;
        }

        .control-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 16px;
          height: 16px;
          background: linear-gradient(135deg, #f5d76e, #d4af37);
          border-radius: 50%;
          cursor: pointer;
          box-shadow: 0 0 10px rgba(212, 175, 55, 0.5);
        }

        .control-slider::-webkit-slider-runnable-track {
          background: linear-gradient(90deg, rgba(139, 105, 20, 0.3), rgba(212, 175, 55, 0.5));
          height: 4px;
          border-radius: 2px;
        }

        .btn-primary {
          padding: 12px 25px;
          background: linear-gradient(135deg, rgba(212, 175, 55, 0.3), rgba(139, 105, 20, 0.3));
          border: 1px solid rgba(212, 175, 55, 0.5);
          border-radius: 30px;
          color: #d4af37;
          font-family: "Cinzel", serif;
          font-size: 0.85rem;
          letter-spacing: 0.08em;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .btn-primary:hover {
          background: linear-gradient(135deg, rgba(212, 175, 55, 0.4), rgba(139, 105, 20, 0.4));
          transform: translateY(-1px);
        }

        .btn-secondary {
          padding: 12px 25px;
          background: transparent;
          border: 1px solid rgba(212, 175, 55, 0.4);
          border-radius: 30px;
          color: rgba(212, 175, 55, 0.8);
          font-family: "Cinzel", serif;
          font-size: 0.85rem;
          letter-spacing: 0.08em;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .btn-ai {
          padding: 12px 25px;
          background: linear-gradient(135deg, rgba(100, 150, 255, 0.3), rgba(75, 100, 200, 0.3));
          border: 1px solid rgba(100, 150, 255, 0.5);
          border-radius: 30px;
          color: #a0c4ff;
          font-family: "Cinzel", serif;
          font-size: 0.85rem;
          letter-spacing: 0.08em;
          cursor: pointer;
          transition: all 0.3s ease;
        }

        .btn-ai:hover {
          background: linear-gradient(135deg, rgba(100, 150, 255, 0.4), rgba(75, 100, 200, 0.4));
        }
      `}</style>

      <StarField />

      <div style={{
        position: 'absolute',
        top: '20%',
        left: '10%',
        width: 300,
        height: 300,
        background: 'radial-gradient(circle, rgba(139, 105, 20, 0.15) 0%, transparent 70%)',
        borderRadius: '50%',
        filter: 'blur(40px)',
        pointerEvents: 'none',
      }} />
      <div style={{
        position: 'absolute',
        bottom: '10%',
        right: '15%',
        width: 400,
        height: 400,
        background: 'radial-gradient(circle, rgba(75, 50, 100, 0.2) 0%, transparent 70%)',
        borderRadius: '50%',
        filter: 'blur(60px)',
        pointerEvents: 'none',
      }} />

      <div style={{
        maxWidth: 1200,
        margin: '0 auto',
        padding: '40px 20px',
        position: 'relative',
        zIndex: 1,
      }}>
        {/* Header */}
        <header style={{ textAlign: 'center', marginBottom: 40 }}>
          <div style={{
            display: 'inline-block',
            marginBottom: 15,
            animation: 'float 6s ease-in-out infinite',
          }}>
            <svg width="50" height="50" viewBox="0 0 100 100" fill="none">
              <circle cx="50" cy="50" r="45" stroke="url(#goldGrad)" strokeWidth="2" fill="none" opacity="0.6"/>
              <circle cx="50" cy="50" r="35" stroke="url(#goldGrad)" strokeWidth="1" fill="none" opacity="0.4"/>
              <path d="M50 20 L55 40 L75 40 L60 52 L65 72 L50 60 L35 72 L40 52 L25 40 L45 40 Z" fill="url(#goldGrad)" opacity="0.8"/>
              <defs>
                <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#8b6914"/>
                  <stop offset="50%" stopColor="#f5d76e"/>
                  <stop offset="100%" stopColor="#8b6914"/>
                </linearGradient>
              </defs>
            </svg>
          </div>

          <h1 style={{
            fontFamily: '"Cinzel", serif',
            fontSize: 'clamp(1.8rem, 4vw, 3rem)',
            fontWeight: 400,
            letterSpacing: '0.15em',
            background: 'linear-gradient(135deg, #8b6914 0%, #f5d76e 30%, #d4af37 50%, #f5d76e 70%, #8b6914 100%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: 10,
          }}>
            PALM ORACLE
          </h1>

          <p style={{
            fontSize: '1rem',
            fontWeight: 300,
            letterSpacing: '0.08em',
            color: 'rgba(232, 220, 200, 0.7)',
          }}>
            AI-Powered Palm Reading & Line Analysis
          </p>
        </header>

        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {!image ? (
          <div
            className="upload-zone mystical-border"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            style={{
              border: '2px dashed rgba(212, 175, 55, 0.4)',
              borderRadius: 20,
              padding: '60px 40px',
              textAlign: 'center',
              cursor: 'pointer',
              background: 'rgba(20, 15, 30, 0.6)',
              backdropFilter: 'blur(10px)',
              transition: 'all 0.4s ease',
              maxWidth: 600,
              margin: '0 auto',
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />

            <div style={{ marginBottom: 25 }}>
              <svg width="70" height="70" viewBox="0 0 100 100" fill="none" style={{ opacity: 0.7 }}>
                <path d="M50 10 C30 10 15 30 15 50 C15 75 30 90 50 90 C70 90 85 75 85 50 C85 30 70 10 50 10"
                      stroke="#d4af37" strokeWidth="2" fill="none"/>
                <path d="M25 50 Q35 35 50 40 Q65 45 75 50" stroke="#d4af37" strokeWidth="1.5" fill="none" opacity="0.6"/>
                <path d="M30 65 Q45 55 50 60 Q55 65 70 55" stroke="#d4af37" strokeWidth="1.5" fill="none" opacity="0.6"/>
                <path d="M50 25 Q48 45 50 70" stroke="#d4af37" strokeWidth="1.5" fill="none" opacity="0.6"/>
              </svg>
            </div>

            <h3 style={{
              fontFamily: '"Cinzel", serif',
              fontSize: '1.2rem',
              fontWeight: 400,
              letterSpacing: '0.1em',
              color: '#d4af37',
              marginBottom: 12,
            }}>
              Present Your Palm
            </h3>

            <p style={{
              color: 'rgba(232, 220, 200, 0.6)',
              fontSize: '0.9rem',
              lineHeight: 1.6,
            }}>
              Drop an image here or click to select<br/>
              <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>For best results, use a clear, well-lit photo</span>
            </p>
          </div>
        ) : (
          <div style={{
            display: 'grid',
            gridTemplateColumns: showReading ? '1fr 1fr' : '1fr',
            gap: 30,
            alignItems: 'start',
          }}>
            {/* Left Column - Image and Controls */}
            <div>
              <div className="mystical-border" style={{
                borderRadius: 20,
                overflow: 'hidden',
                background: 'rgba(20, 15, 30, 0.8)',
                backdropFilter: 'blur(10px)',
              }}>
                <div style={{
                  position: 'relative',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  minHeight: 350,
                  padding: 15,
                  background: 'rgba(10, 10, 18, 0.5)',
                }}>
                  {(isProcessing || vlmProcessing) && (
                    <div style={{
                      position: 'absolute',
                      inset: 0,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      background: 'rgba(10, 10, 18, 0.9)',
                      zIndex: 10,
                    }}>
                      <div style={{
                        width: 50,
                        height: 50,
                        border: '3px solid rgba(212, 175, 55, 0.2)',
                        borderTopColor: vlmProcessing ? '#a0c4ff' : '#d4af37',
                        borderRadius: '50%',
                        animation: 'spin 1s linear infinite',
                      }} />
                      <p style={{
                        marginTop: 15,
                        fontFamily: '"Cinzel", serif',
                        letterSpacing: '0.1em',
                        color: vlmProcessing ? '#a0c4ff' : '#d4af37',
                      }}>
                        {vlmProcessing ? 'AI analyzing palm lines...' : 'Processing...'}
                      </p>
                    </div>
                  )}

                  <div style={{ position: 'relative', display: 'inline-block' }}>
                    <img
                      ref={displayedImageRef}
                      src={image}
                      alt="Palm"
                      onLoad={updateDisplayedDimensions}
                      style={{
                        maxWidth: '100%',
                        maxHeight: 450,
                        borderRadius: 10,
                        display: 'block',
                      }}
                    />

                    {processedImage && showOverlay && !showOriginal && (
                      <img
                        key={animationKey}
                        src={processedImage}
                        alt="Palm lines overlay"
                        style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '100%',
                          borderRadius: 10,
                          animation: 'revealLines 1.5s ease-out forwards',
                          mixBlendMode: 'screen',
                        }}
                      />
                    )}

                    {/* AI Line Annotations */}
                    {vlmEnabled && showLineAnnotations && vlmResult && !showOriginal && displayedDimensions.width > 0 && (
                      <LineAnnotationCanvas
                        vlmResult={vlmResult}
                        width={displayedDimensions.width}
                        height={displayedDimensions.height}
                      />
                    )}
                  </div>
                </div>

                {/* Controls */}
                <div style={{
                  padding: '20px',
                  borderTop: '1px solid rgba(212, 175, 55, 0.2)',
                  background: 'rgba(15, 12, 25, 0.5)',
                }}>
                  {/* Main Buttons */}
                  <div style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: 10,
                    marginBottom: 20,
                    justifyContent: 'center',
                  }}>
                    <button
                      className={showOverlay ? 'btn-primary' : 'btn-secondary'}
                      onClick={() => setShowOverlay(!showOverlay)}
                    >
                      {showOverlay ? '✧ Lines Visible' : '○ Lines Hidden'}
                    </button>

                    <button
                      className="btn-secondary"
                      onMouseDown={() => setShowOriginal(true)}
                      onMouseUp={() => setShowOriginal(false)}
                      onMouseLeave={() => setShowOriginal(false)}
                      onTouchStart={() => setShowOriginal(true)}
                      onTouchEnd={() => setShowOriginal(false)}
                    >
                      Hold for Original
                    </button>

                    <button
                      className={vlmEnabled ? 'btn-ai' : 'btn-secondary'}
                      onClick={() => setVlmEnabled(!vlmEnabled)}
                      style={{
                        borderColor: 'rgba(100, 150, 255, 0.5)',
                        color: vlmEnabled ? '#a0c4ff' : 'rgba(160, 196, 255, 0.7)',
                      }}
                    >
                      {vlmEnabled ? '✦ AI Vision On' : '○ AI Vision'}
                    </button>

                    <button
                      className="btn-secondary"
                      onClick={() => setShowSettings(!showSettings)}
                      style={{ padding: '12px 18px' }}
                    >
                      ⚙
                    </button>

                    <button
                      className="btn-secondary"
                      onClick={replayAnimation}
                      title="Replay Animation"
                    >
                      Replay
                    </button>

                    <button
                      className="btn-secondary"
                      onClick={downloadResult}
                      title="Download Result"
                    >
                      Download
                    </button>
                  </div>

                  {/* Error Display */}
                  {vlmError && (
                    <div style={{
                      marginBottom: 15,
                      padding: '10px 15px',
                      background: 'rgba(200, 100, 100, 0.15)',
                      borderRadius: 8,
                      border: '1px solid rgba(200, 100, 100, 0.3)',
                      fontSize: '0.85rem',
                      color: '#ffb0b0',
                      textAlign: 'center',
                    }}>
                      {vlmError}
                    </div>
                  )}

                  {/* Settings Panel */}
                  {showSettings && (
                    <div style={{
                      marginBottom: 20,
                      padding: '15px',
                      background: 'rgba(30, 25, 45, 0.8)',
                      borderRadius: 10,
                      border: '1px solid rgba(212, 175, 55, 0.2)',
                    }}>
                      <h4 style={{
                        fontFamily: '"Cinzel", serif',
                        fontSize: '0.9rem',
                        color: '#d4af37',
                        marginTop: 0,
                        marginBottom: 12,
                      }}>
                        AI Settings (Anthropic Claude)
                      </h4>

                      <div style={{ marginBottom: 12 }}>
                        <label style={{ display: 'block', marginBottom: 5, fontSize: '0.8rem', color: 'rgba(232, 220, 200, 0.7)' }}>
                          Anthropic API Key
                        </label>
                        <input
                          type="password"
                          value={apiKey}
                          onChange={(e) => setApiKey(e.target.value)}
                          placeholder="Enter Anthropic API key"
                          style={{
                            width: '100%',
                            padding: '8px 12px',
                            background: 'rgba(20, 15, 30, 0.8)',
                            border: '1px solid rgba(212, 175, 55, 0.3)',
                            borderRadius: 6,
                            color: '#e8dcc8',
                            fontSize: '0.85rem',
                            boxSizing: 'border-box',
                          }}
                        />
                      </div>

                      {vlmEnabled && apiKey && (
                        <button
                          className="btn-ai"
                          onClick={() => runVLMAnalysis(image)}
                          disabled={vlmProcessing}
                          style={{ width: '100%', marginTop: 5 }}
                        >
                          {vlmProcessing ? 'Analyzing...' : 'Re-analyze Palm'}
                        </button>
                      )}
                    </div>
                  )}

                  {/* Detected Lines Display */}
                  {vlmEnabled && vlmResult && vlmResult.lines && vlmResult.lines.length > 0 && (
                    <div style={{
                      marginBottom: 20,
                      padding: '15px',
                      background: 'rgba(100, 150, 255, 0.08)',
                      borderRadius: 10,
                      border: '1px solid rgba(100, 150, 255, 0.2)',
                    }}>
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: 10,
                      }}>
                        <span style={{ fontSize: '0.85rem', color: '#a0c4ff', fontFamily: '"Cinzel", serif' }}>
                          Detected Lines
                        </span>
                        <button
                          onClick={() => setShowLineAnnotations(!showLineAnnotations)}
                          style={{
                            padding: '4px 10px',
                            background: showLineAnnotations ? 'rgba(100, 150, 255, 0.2)' : 'transparent',
                            border: '1px solid rgba(100, 150, 255, 0.3)',
                            borderRadius: 12,
                            color: '#a0c4ff',
                            fontSize: '0.7rem',
                            cursor: 'pointer',
                          }}
                        >
                          {showLineAnnotations ? 'Annotations On' : 'Annotations Off'}
                        </button>
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                        {vlmResult.lines.map((line, idx) => {
                          const color = getLineColor(line.name);
                          return (
                            <span
                              key={idx}
                              style={{
                                padding: '3px 10px',
                                background: `rgba(${color.r}, ${color.g}, ${color.b}, 0.2)`,
                                border: `1px solid ${color.hex}`,
                                borderRadius: 12,
                                fontSize: '0.75rem',
                                color: color.hex,
                              }}
                            >
                              {line.name}
                            </span>
                          );
                        })}
                      </div>

                      {/* Generate Reading Button */}
                      <button
                        className="btn-primary"
                        onClick={generateReading}
                        disabled={isGeneratingReading}
                        style={{
                          width: '100%',
                          marginTop: 15,
                          background: 'linear-gradient(135deg, rgba(212, 175, 55, 0.4), rgba(139, 105, 20, 0.4))',
                        }}
                      >
                        {isGeneratingReading ? 'Generating Reading...' : '✧ Generate Palm Reading ✧'}
                      </button>
                    </div>
                  )}

                  {/* Sliders */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 15 }}>
                    <div>
                      <label style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: 8,
                        fontSize: '0.8rem',
                        color: 'rgba(232, 220, 200, 0.7)',
                      }}>
                        <span>Sensitivity</span>
                        <span style={{ color: '#d4af37' }}>{sensitivity}%</span>
                      </label>
                      <input
                        type="range"
                        min="20"
                        max="95"
                        value={sensitivity}
                        onChange={(e) => setSensitivity(Number(e.target.value))}
                        className="control-slider"
                        style={{
                          width: '100%',
                          height: 20,
                          background: 'transparent',
                          cursor: 'pointer',
                          WebkitAppearance: 'none',
                        }}
                      />
                    </div>

                    <div>
                      <label style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: 8,
                        fontSize: '0.8rem',
                        color: 'rgba(232, 220, 200, 0.7)',
                      }}>
                        <span>Thickness</span>
                        <span style={{ color: '#d4af37' }}>{lineThickness}px</span>
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="5"
                        value={lineThickness}
                        onChange={(e) => setLineThickness(Number(e.target.value))}
                        className="control-slider"
                        style={{
                          width: '100%',
                          height: 20,
                          background: 'transparent',
                          cursor: 'pointer',
                          WebkitAppearance: 'none',
                        }}
                      />
                    </div>
                  </div>

                  {/* New Reading Button */}
                  <button
                    onClick={() => {
                      setImage(null);
                      setProcessedImage(null);
                      setVlmResult(null);
                      setVlmMask(null);
                      setVlmError(null);
                      setPalmReading(null);
                      setShowReading(false);
                    }}
                    style={{
                      width: '100%',
                      marginTop: 15,
                      padding: '10px',
                      background: 'transparent',
                      border: '1px solid rgba(150, 100, 100, 0.4)',
                      borderRadius: 8,
                      color: 'rgba(200, 150, 150, 0.8)',
                      fontFamily: '"Cinzel", serif',
                      fontSize: '0.8rem',
                      cursor: 'pointer',
                    }}
                  >
                    New Reading
                  </button>
                </div>
              </div>
            </div>

            {/* Right Column - Reading */}
            {showReading && (
              <div className="mystical-border" style={{
                borderRadius: 20,
                overflow: 'hidden',
                background: 'rgba(20, 15, 30, 0.8)',
                backdropFilter: 'blur(10px)',
                maxHeight: '80vh',
                overflowY: 'auto',
              }}>
                <div style={{
                  padding: '20px 25px',
                  borderBottom: '1px solid rgba(212, 175, 55, 0.2)',
                  background: 'rgba(15, 12, 25, 0.5)',
                  position: 'sticky',
                  top: 0,
                  zIndex: 5,
                }}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}>
                    <h3 style={{
                      fontFamily: '"Cinzel", serif',
                      fontSize: '1.1rem',
                      letterSpacing: '0.1em',
                      color: '#d4af37',
                      margin: 0,
                    }}>
                      ✧ Your Palm Reading ✧
                    </h3>
                    <button
                      onClick={() => setShowReading(false)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        color: 'rgba(232, 220, 200, 0.5)',
                        fontSize: '1.2rem',
                        cursor: 'pointer',
                      }}
                    >
                      ✕
                    </button>
                  </div>
                </div>

                <ReadingSection reading={palmReading} isGenerating={isGeneratingReading} />
              </div>
            )}
          </div>
        )}

        {/* Privacy Notice */}
        <div style={{
          marginTop: 30,
          padding: '15px 20px',
          background: 'rgba(20, 15, 30, 0.6)',
          borderRadius: 10,
          border: '1px solid rgba(212, 175, 55, 0.15)',
          maxWidth: 600,
          margin: '30px auto 0',
        }}>
          <p style={{
            fontSize: '0.8rem',
            color: 'rgba(232, 220, 200, 0.6)',
            textAlign: 'center',
            margin: 0,
            lineHeight: 1.6,
          }}>
            <strong style={{ color: 'rgba(212, 175, 55, 0.8)' }}>Privacy:</strong> Your palm images are processed entirely within your browser.
            No images are uploaded to any server. Your data remains private and secure on your device.
          </p>
        </div>

        {/* Footer */}
        <footer style={{
          marginTop: 20,
          textAlign: 'center',
          fontSize: '0.75rem',
          color: 'rgba(232, 220, 200, 0.4)',
          letterSpacing: '0.05em',
        }}>
          <p>For entertainment and spiritual exploration purposes</p>
        </footer>
      </div>
    </div>
  );
}
