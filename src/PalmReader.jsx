import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  analyzeWithVLM,
  generateVLMGuidedMask,
  combineEdgeDetectionWithVLM,
  LINE_COLORS,
  VLM_PROVIDERS,
} from './vlmService';

// Palm line detection using edge detection and line enhancement
const detectPalmLines = (imageData, sensitivity = 50, lineThickness = 2, vlmMask = null, vlmWeight = 0.5) => {
  const { data, width, height } = imageData;
  const gray = new Float32Array(width * height);
  const output = new Uint8ClampedArray(data.length);

  // Convert to grayscale with enhanced contrast
  for (let i = 0; i < data.length; i += 4) {
    const idx = i / 4;
    gray[idx] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  // Apply Gaussian blur to reduce noise
  const blurred = new Float32Array(width * height);
  const kernel = [1, 4, 6, 4, 1];
  const kernelSum = 16;

  // Horizontal pass
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -2; k <= 2; k++) {
        const px = Math.min(Math.max(x + k, 0), width - 1);
        sum += gray[y * width + px] * kernel[k + 2];
      }
      blurred[y * width + x] = sum / kernelSum;
    }
  }

  // Vertical pass
  const blurred2 = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -2; k <= 2; k++) {
        const py = Math.min(Math.max(y + k, 0), height - 1);
        sum += blurred[py * width + x] * kernel[k + 2];
      }
      blurred2[y * width + x] = sum / kernelSum;
    }
  }

  // Sobel edge detection
  const edges = new Float32Array(width * height);
  const directions = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;

      // Sobel kernels
      const gx = (
        -blurred2[(y - 1) * width + (x - 1)] - 2 * blurred2[y * width + (x - 1)] - blurred2[(y + 1) * width + (x - 1)] +
        blurred2[(y - 1) * width + (x + 1)] + 2 * blurred2[y * width + (x + 1)] + blurred2[(y + 1) * width + (x + 1)]
      );

      const gy = (
        -blurred2[(y - 1) * width + (x - 1)] - 2 * blurred2[(y - 1) * width + x] - blurred2[(y - 1) * width + (x + 1)] +
        blurred2[(y + 1) * width + (x - 1)] + 2 * blurred2[(y + 1) * width + x] + blurred2[(y + 1) * width + (x + 1)]
      );

      edges[idx] = Math.sqrt(gx * gx + gy * gy);
      directions[idx] = Math.atan2(gy, gx);
    }
  }

  // Non-maximum suppression for thinner lines
  const suppressed = new Float32Array(width * height);

  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      const idx = y * width + x;
      const angle = directions[idx];
      const mag = edges[idx];

      let neighbor1, neighbor2;

      // Round angle to nearest 45 degrees
      const sector = Math.round(((angle + Math.PI) / Math.PI) * 4) % 4;

      switch (sector) {
        case 0: // Horizontal
          neighbor1 = edges[y * width + (x - 1)];
          neighbor2 = edges[y * width + (x + 1)];
          break;
        case 1: // Diagonal /
          neighbor1 = edges[(y - 1) * width + (x + 1)];
          neighbor2 = edges[(y + 1) * width + (x - 1)];
          break;
        case 2: // Vertical
          neighbor1 = edges[(y - 1) * width + x];
          neighbor2 = edges[(y + 1) * width + x];
          break;
        case 3: // Diagonal \
          neighbor1 = edges[(y - 1) * width + (x - 1)];
          neighbor2 = edges[(y + 1) * width + (x + 1)];
          break;
        default:
          neighbor1 = neighbor2 = 0;
      }

      if (mag >= neighbor1 && mag >= neighbor2) {
        suppressed[idx] = mag;
      }
    }
  }

  // Find max edge value for thresholding
  let maxEdge = 0;
  for (let i = 0; i < suppressed.length; i++) {
    if (suppressed[i] > maxEdge) maxEdge = suppressed[i];
  }

  // Adaptive thresholding based on sensitivity
  const threshold = maxEdge * (1 - sensitivity / 100) * 0.15;
  const lowThreshold = threshold * 0.4;

  // Hysteresis thresholding with line enhancement
  const result = new Uint8Array(width * height);

  // VLM-enhanced thresholding: lower threshold in VLM-detected areas
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;

      // If VLM mask provided, use adaptive threshold based on VLM guidance
      let effectiveThreshold = threshold;
      let effectiveLowThreshold = lowThreshold;

      if (vlmMask && vlmMask[idx] > 0) {
        // Lower threshold significantly in VLM-detected line areas
        const vlmFactor = 1 - (vlmWeight * 0.7);
        effectiveThreshold = threshold * vlmFactor;
        effectiveLowThreshold = lowThreshold * vlmFactor;
      }

      if (suppressed[idx] > effectiveThreshold) {
        result[idx] = 255;
      } else if (suppressed[idx] > effectiveLowThreshold) {
        // Check if connected to strong edge
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

        // Also include if in VLM-detected area and has some edge strength
        if (!connected && vlmMask && vlmMask[idx] > 0 && suppressed[idx] > lowThreshold * 0.3) {
          result[idx] = 255;
        }
      }
    }
  }

  // Dilate lines for visibility based on thickness setting
  const dilated = new Uint8Array(width * height);
  const dilateRadius = Math.max(1, Math.floor(lineThickness / 2));

  for (let y = dilateRadius; y < height - dilateRadius; y++) {
    for (let x = dilateRadius; x < width - dilateRadius; x++) {
      if (result[y * width + x] === 255) {
        for (let dy = -dilateRadius; dy <= dilateRadius; dy++) {
          for (let dx = -dilateRadius; dx <= dilateRadius; dx++) {
            if (dx * dx + dy * dy <= dilateRadius * dilateRadius + 1) {
              dilated[(y + dy) * width + (x + dx)] = 255;
            }
          }
        }
      }
    }
  }

  // Create output with golden lines on transparent background
  for (let i = 0; i < data.length; i += 4) {
    const idx = i / 4;
    if (dilated[idx] === 255) {
      // Golden color for lines
      output[i] = 255;     // R
      output[i + 1] = 200; // G
      output[i + 2] = 100; // B
      output[i + 3] = 255; // A
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

export default function PalmReader() {
  const [image, setImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [sensitivity, setSensitivity] = useState(60);
  const [lineThickness, setLineThickness] = useState(2);
  const [showOriginal, setShowOriginal] = useState(false);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // VLM-related state
  const [vlmEnabled, setVlmEnabled] = useState(false);
  const [vlmProvider, setVlmProvider] = useState(VLM_PROVIDERS.OPENAI);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('vlm_api_key') || '');
  const [vlmResult, setVlmResult] = useState(null);
  const [vlmMask, setVlmMask] = useState(null);
  const [vlmWeight, setVlmWeight] = useState(0.6);
  const [vlmProcessing, setVlmProcessing] = useState(false);
  const [vlmError, setVlmError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showLineLabels, setShowLineLabels] = useState(true);

  // Save API key to localStorage when changed
  useEffect(() => {
    if (apiKey) {
      localStorage.setItem('vlm_api_key', apiKey);
    }
  }, [apiKey]);

  const processImage = useCallback((imgSrc, sens = sensitivity, thickness = lineThickness, currentVlmMask = vlmMask, currentVlmWeight = vlmWeight) => {
    setIsProcessing(true);

    const img = new Image();
    img.onload = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      // Scale image to reasonable size for processing
      const maxSize = 800;
      let { width, height } = img;
      if (width > maxSize || height > maxSize) {
        const ratio = Math.min(maxSize / width, maxSize / height);
        width *= ratio;
        height *= ratio;
      }

      canvas.width = width;
      canvas.height = height;
      ctx.drawImage(img, 0, 0, width, height);

      const imageData = ctx.getImageData(0, 0, width, height);

      // Pass VLM mask if available and VLM is enabled
      const maskToUse = vlmEnabled && currentVlmMask ? currentVlmMask : null;
      const processed = detectPalmLines(imageData, sens, thickness, maskToUse, currentVlmWeight);

      // Create processed image URL
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

  // VLM analysis function
  const runVLMAnalysis = useCallback(async (imgSrc) => {
    if (!apiKey) {
      setVlmError('Please enter an API key in settings');
      setShowSettings(true);
      return;
    }

    setVlmProcessing(true);
    setVlmError(null);

    try {
      const result = await analyzeWithVLM(imgSrc, vlmProvider, apiKey);
      setVlmResult(result);

      // Generate VLM-guided mask if we have line data
      if (result.lines && result.lines.length > 0) {
        // Get image dimensions
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

          // Reprocess image with VLM guidance
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
  }, [apiKey, vlmProvider, sensitivity, lineThickness, vlmWeight, processImage]);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        processImage(event.target.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
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

  // Rerun VLM analysis when VLM is enabled and we have an image
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
      `}</style>

      <StarField />

      {/* Ambient glow effects */}
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
        maxWidth: 1000,
        margin: '0 auto',
        padding: '40px 20px',
        position: 'relative',
        zIndex: 1,
      }}>
        {/* Header */}
        <header style={{ textAlign: 'center', marginBottom: 50 }}>
          <div style={{
            display: 'inline-block',
            marginBottom: 20,
            animation: 'float 6s ease-in-out infinite',
          }}>
            <svg width="60" height="60" viewBox="0 0 100 100" fill="none">
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
            fontSize: 'clamp(2rem, 5vw, 3.5rem)',
            fontWeight: 400,
            letterSpacing: '0.15em',
            background: 'linear-gradient(135deg, #8b6914 0%, #f5d76e 30%, #d4af37 50%, #f5d76e 70%, #8b6914 100%)',
            backgroundClip: 'text',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            marginBottom: 15,
            textShadow: '0 0 30px rgba(212, 175, 55, 0.3)',
          }}>
            PALM ORACLE
          </h1>

          <p style={{
            fontSize: '1.1rem',
            fontWeight: 300,
            letterSpacing: '0.1em',
            color: 'rgba(232, 220, 200, 0.7)',
            maxWidth: 500,
            margin: '0 auto',
            lineHeight: 1.8,
          }}>
            Reveal the sacred lines inscribed upon your palm
          </p>
        </header>

        {/* Hidden canvas for processing */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {/* Upload Area */}
        {!image ? (
          <div
            className="upload-zone mystical-border"
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            style={{
              border: '2px dashed rgba(212, 175, 55, 0.4)',
              borderRadius: 20,
              padding: '80px 40px',
              textAlign: 'center',
              cursor: 'pointer',
              background: 'rgba(20, 15, 30, 0.6)',
              backdropFilter: 'blur(10px)',
              transition: 'all 0.4s ease',
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />

            <div style={{ marginBottom: 30 }}>
              <svg width="80" height="80" viewBox="0 0 100 100" fill="none" style={{ opacity: 0.7 }}>
                <path d="M50 10 C30 10 15 30 15 50 C15 75 30 90 50 90 C70 90 85 75 85 50 C85 30 70 10 50 10"
                      stroke="#d4af37" strokeWidth="2" fill="none"/>
                <path d="M25 50 Q35 35 50 40 Q65 45 75 50" stroke="#d4af37" strokeWidth="1.5" fill="none" opacity="0.6"/>
                <path d="M30 65 Q45 55 50 60 Q55 65 70 55" stroke="#d4af37" strokeWidth="1.5" fill="none" opacity="0.6"/>
                <path d="M50 25 Q48 45 50 70" stroke="#d4af37" strokeWidth="1.5" fill="none" opacity="0.6"/>
                <circle cx="40" cy="45" r="3" fill="#d4af37" opacity="0.5"/>
                <circle cx="60" cy="48" r="3" fill="#d4af37" opacity="0.5"/>
                <circle cx="50" cy="60" r="3" fill="#d4af37" opacity="0.5"/>
              </svg>
            </div>

            <h3 style={{
              fontFamily: '"Cinzel", serif',
              fontSize: '1.3rem',
              fontWeight: 400,
              letterSpacing: '0.1em',
              color: '#d4af37',
              marginBottom: 15,
            }}>
              Present Your Palm
            </h3>

            <p style={{
              color: 'rgba(232, 220, 200, 0.6)',
              fontSize: '0.95rem',
              lineHeight: 1.6,
            }}>
              Drop an image here or click to select<br/>
              <span style={{ fontSize: '0.85rem', opacity: 0.7 }}>For best results, use a clear, well-lit photo of your palm</span>
            </p>
          </div>
        ) : (
          <>
            {/* Image Display */}
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
                minHeight: 400,
                padding: 20,
              }}>
                {isProcessing && (
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
                      width: 60,
                      height: 60,
                      border: '3px solid rgba(212, 175, 55, 0.2)',
                      borderTopColor: '#d4af37',
                      borderRadius: '50%',
                      animation: 'spin 1s linear infinite',
                    }} />
                    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                    <p style={{
                      marginTop: 20,
                      fontFamily: '"Cinzel", serif',
                      letterSpacing: '0.1em',
                      color: '#d4af37',
                      animation: 'pulse 2s ease-in-out infinite',
                    }}>
                      Reading the lines...
                    </p>
                  </div>
                )}

                <div style={{ position: 'relative', display: 'inline-block' }}>
                  <img
                    src={image}
                    alt="Palm"
                    style={{
                      maxWidth: '100%',
                      maxHeight: 500,
                      borderRadius: 10,
                      display: 'block',
                    }}
                  />

                  {processedImage && showOverlay && !showOriginal && (
                    <img
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

                  {/* VLM Line Labels Overlay */}
                  {vlmEnabled && showLineLabels && vlmResult && vlmResult.lines && vlmResult.lines.length > 0 && !showOriginal && (
                    <div style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      pointerEvents: 'none',
                    }}>
                      {vlmResult.lines.map((line, idx) => {
                        const color = LINE_COLORS[line.name] || LINE_COLORS.default;
                        // Position label at the midpoint of the line
                        const labelX = line.startX !== undefined && line.endX !== undefined
                          ? (line.startX + line.endX) / 2
                          : 50;
                        const labelY = line.startY !== undefined && line.endY !== undefined
                          ? Math.min(line.startY, line.endY) - 5
                          : 50;

                        return (
                          <div
                            key={idx}
                            style={{
                              position: 'absolute',
                              left: `${labelX}%`,
                              top: `${labelY}%`,
                              transform: 'translate(-50%, -100%)',
                              padding: '3px 8px',
                              background: `rgba(${color.r}, ${color.g}, ${color.b}, 0.85)`,
                              borderRadius: 4,
                              fontSize: '0.65rem',
                              fontWeight: 600,
                              color: '#fff',
                              textShadow: '0 1px 2px rgba(0,0,0,0.5)',
                              whiteSpace: 'nowrap',
                              zIndex: 10,
                            }}
                          >
                            {line.name}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>

              {/* Controls */}
              <div style={{
                padding: '25px 30px',
                borderTop: '1px solid rgba(212, 175, 55, 0.2)',
                background: 'rgba(15, 12, 25, 0.5)',
              }}>
                {/* Toggle Controls */}
                <div style={{
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 15,
                  marginBottom: 25,
                  justifyContent: 'center',
                }}>
                  <button
                    onClick={() => setShowOverlay(!showOverlay)}
                    style={{
                      padding: '12px 25px',
                      background: showOverlay
                        ? 'linear-gradient(135deg, rgba(212, 175, 55, 0.3), rgba(139, 105, 20, 0.3))'
                        : 'transparent',
                      border: '1px solid rgba(212, 175, 55, 0.5)',
                      borderRadius: 30,
                      color: '#d4af37',
                      fontFamily: '"Cinzel", serif',
                      fontSize: '0.85rem',
                      letterSpacing: '0.08em',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                    }}
                  >
                    {showOverlay ? '✧ Lines Visible' : '○ Lines Hidden'}
                  </button>

                  <button
                    onMouseDown={() => setShowOriginal(true)}
                    onMouseUp={() => setShowOriginal(false)}
                    onMouseLeave={() => setShowOriginal(false)}
                    onTouchStart={() => setShowOriginal(true)}
                    onTouchEnd={() => setShowOriginal(false)}
                    style={{
                      padding: '12px 25px',
                      background: 'transparent',
                      border: '1px solid rgba(212, 175, 55, 0.5)',
                      borderRadius: 30,
                      color: '#d4af37',
                      fontFamily: '"Cinzel", serif',
                      fontSize: '0.85rem',
                      letterSpacing: '0.08em',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                    }}
                  >
                    Hold to View Original
                  </button>

                  <button
                    onClick={() => {
                      setImage(null);
                      setProcessedImage(null);
                      setVlmResult(null);
                      setVlmMask(null);
                      setVlmError(null);
                    }}
                    style={{
                      padding: '12px 25px',
                      background: 'transparent',
                      border: '1px solid rgba(150, 100, 100, 0.5)',
                      borderRadius: 30,
                      color: 'rgba(200, 150, 150, 0.8)',
                      fontFamily: '"Cinzel", serif',
                      fontSize: '0.85rem',
                      letterSpacing: '0.08em',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                    }}
                  >
                    New Reading
                  </button>

                  <button
                    onClick={() => setVlmEnabled(!vlmEnabled)}
                    style={{
                      padding: '12px 25px',
                      background: vlmEnabled
                        ? 'linear-gradient(135deg, rgba(100, 150, 255, 0.3), rgba(75, 100, 200, 0.3))'
                        : 'transparent',
                      border: '1px solid rgba(100, 150, 255, 0.5)',
                      borderRadius: 30,
                      color: vlmEnabled ? '#a0c4ff' : 'rgba(160, 196, 255, 0.7)',
                      fontFamily: '"Cinzel", serif',
                      fontSize: '0.85rem',
                      letterSpacing: '0.08em',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                    }}
                  >
                    {vlmEnabled ? '✦ AI Vision On' : '○ AI Vision Off'}
                  </button>

                  <button
                    onClick={() => setShowSettings(!showSettings)}
                    style={{
                      padding: '12px 20px',
                      background: 'transparent',
                      border: '1px solid rgba(212, 175, 55, 0.3)',
                      borderRadius: 30,
                      color: 'rgba(212, 175, 55, 0.7)',
                      fontFamily: '"Cinzel", serif',
                      fontSize: '0.85rem',
                      letterSpacing: '0.08em',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                    }}
                  >
                    Settings
                  </button>
                </div>

                {/* VLM Status/Error */}
                {vlmEnabled && (vlmProcessing || vlmError) && (
                  <div style={{
                    marginBottom: 20,
                    padding: '12px 20px',
                    background: vlmError ? 'rgba(200, 100, 100, 0.15)' : 'rgba(100, 150, 255, 0.15)',
                    borderRadius: 10,
                    border: `1px solid ${vlmError ? 'rgba(200, 100, 100, 0.3)' : 'rgba(100, 150, 255, 0.3)'}`,
                    textAlign: 'center',
                    fontSize: '0.85rem',
                  }}>
                    {vlmProcessing ? (
                      <span style={{ color: '#a0c4ff' }}>Analyzing palm with AI vision model...</span>
                    ) : vlmError ? (
                      <span style={{ color: '#ffb0b0' }}>{vlmError}</span>
                    ) : null}
                  </div>
                )}

                {/* VLM Detected Lines */}
                {vlmEnabled && vlmResult && vlmResult.lines && vlmResult.lines.length > 0 && (
                  <div style={{
                    marginBottom: 20,
                    padding: '15px 20px',
                    background: 'rgba(100, 150, 255, 0.1)',
                    borderRadius: 12,
                    border: '1px solid rgba(100, 150, 255, 0.2)',
                  }}>
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: 12,
                    }}>
                      <h4 style={{
                        fontFamily: '"Cinzel", serif',
                        fontSize: '0.9rem',
                        letterSpacing: '0.08em',
                        color: '#a0c4ff',
                        margin: 0,
                      }}>
                        AI Detected Lines
                      </h4>
                      <button
                        onClick={() => setShowLineLabels(!showLineLabels)}
                        style={{
                          padding: '5px 12px',
                          background: showLineLabels ? 'rgba(100, 150, 255, 0.2)' : 'transparent',
                          border: '1px solid rgba(100, 150, 255, 0.3)',
                          borderRadius: 15,
                          color: '#a0c4ff',
                          fontSize: '0.75rem',
                          cursor: 'pointer',
                        }}
                      >
                        {showLineLabels ? 'Labels On' : 'Labels Off'}
                      </button>
                    </div>
                    <div style={{
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: 8,
                    }}>
                      {vlmResult.lines.map((line, idx) => {
                        const color = LINE_COLORS[line.name] || LINE_COLORS.default;
                        return (
                          <span
                            key={idx}
                            style={{
                              padding: '4px 12px',
                              background: `rgba(${color.r}, ${color.g}, ${color.b}, 0.2)`,
                              border: `1px solid rgba(${color.r}, ${color.g}, ${color.b}, 0.4)`,
                              borderRadius: 15,
                              fontSize: '0.8rem',
                              color: `rgb(${Math.min(255, color.r + 50)}, ${Math.min(255, color.g + 50)}, ${Math.min(255, color.b + 50)})`,
                            }}
                          >
                            {line.name}
                          </span>
                        );
                      })}
                    </div>
                    {vlmResult.palmQuality && (
                      <p style={{
                        marginTop: 10,
                        marginBottom: 0,
                        fontSize: '0.8rem',
                        color: 'rgba(232, 220, 200, 0.6)',
                      }}>
                        Image quality: <span style={{ color: '#a0c4ff' }}>{vlmResult.palmQuality}</span>
                      </p>
                    )}
                  </div>
                )}

                {/* Settings Panel */}
                {showSettings && (
                  <div style={{
                    marginBottom: 25,
                    padding: '20px',
                    background: 'rgba(30, 25, 45, 0.8)',
                    borderRadius: 12,
                    border: '1px solid rgba(212, 175, 55, 0.2)',
                  }}>
                    <h4 style={{
                      fontFamily: '"Cinzel", serif',
                      fontSize: '0.95rem',
                      letterSpacing: '0.08em',
                      color: '#d4af37',
                      marginTop: 0,
                      marginBottom: 15,
                    }}>
                      AI Vision Settings
                    </h4>

                    <div style={{ marginBottom: 15 }}>
                      <label style={{
                        display: 'block',
                        marginBottom: 8,
                        fontSize: '0.85rem',
                        color: 'rgba(232, 220, 200, 0.8)',
                      }}>
                        Provider
                      </label>
                      <select
                        value={vlmProvider}
                        onChange={(e) => setVlmProvider(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '10px 15px',
                          background: 'rgba(20, 15, 30, 0.8)',
                          border: '1px solid rgba(212, 175, 55, 0.3)',
                          borderRadius: 8,
                          color: '#e8dcc8',
                          fontSize: '0.9rem',
                          cursor: 'pointer',
                        }}
                      >
                        <option value={VLM_PROVIDERS.OPENAI}>OpenAI (GPT-4o)</option>
                        <option value={VLM_PROVIDERS.ANTHROPIC}>Anthropic (Claude)</option>
                      </select>
                    </div>

                    <div style={{ marginBottom: 15 }}>
                      <label style={{
                        display: 'block',
                        marginBottom: 8,
                        fontSize: '0.85rem',
                        color: 'rgba(232, 220, 200, 0.8)',
                      }}>
                        API Key
                      </label>
                      <input
                        type="password"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder={`Enter your ${vlmProvider === VLM_PROVIDERS.OPENAI ? 'OpenAI' : 'Anthropic'} API key`}
                        style={{
                          width: '100%',
                          padding: '10px 15px',
                          background: 'rgba(20, 15, 30, 0.8)',
                          border: '1px solid rgba(212, 175, 55, 0.3)',
                          borderRadius: 8,
                          color: '#e8dcc8',
                          fontSize: '0.9rem',
                          boxSizing: 'border-box',
                        }}
                      />
                      <p style={{
                        marginTop: 5,
                        marginBottom: 0,
                        fontSize: '0.75rem',
                        color: 'rgba(232, 220, 200, 0.5)',
                      }}>
                        Your API key is stored locally in your browser
                      </p>
                    </div>

                    {vlmEnabled && apiKey && (
                      <button
                        onClick={() => runVLMAnalysis(image)}
                        disabled={vlmProcessing}
                        style={{
                          width: '100%',
                          padding: '12px 20px',
                          background: vlmProcessing ? 'rgba(100, 150, 255, 0.2)' : 'linear-gradient(135deg, rgba(100, 150, 255, 0.3), rgba(75, 100, 200, 0.3))',
                          border: '1px solid rgba(100, 150, 255, 0.5)',
                          borderRadius: 8,
                          color: '#a0c4ff',
                          fontFamily: '"Cinzel", serif',
                          fontSize: '0.85rem',
                          letterSpacing: '0.05em',
                          cursor: vlmProcessing ? 'wait' : 'pointer',
                          transition: 'all 0.3s ease',
                        }}
                      >
                        {vlmProcessing ? 'Analyzing...' : 'Re-analyze with AI'}
                      </button>
                    )}
                  </div>
                )}

                {/* Sliders */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
                  gap: 25,
                }}>
                  <div>
                    <label style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      marginBottom: 10,
                      fontSize: '0.85rem',
                      letterSpacing: '0.08em',
                      color: 'rgba(232, 220, 200, 0.8)',
                    }}>
                      <span>Line Sensitivity</span>
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
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      fontSize: '0.7rem',
                      color: 'rgba(232, 220, 200, 0.4)',
                      marginTop: 5,
                    }}>
                      <span>Major Lines</span>
                      <span>All Lines</span>
                    </div>
                  </div>

                  <div>
                    <label style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      marginBottom: 10,
                      fontSize: '0.85rem',
                      letterSpacing: '0.08em',
                      color: 'rgba(232, 220, 200, 0.8)',
                    }}>
                      <span>Line Thickness</span>
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
                    <div style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      fontSize: '0.7rem',
                      color: 'rgba(232, 220, 200, 0.4)',
                      marginTop: 5,
                    }}>
                      <span>Fine</span>
                      <span>Bold</span>
                    </div>
                  </div>

                  {/* VLM Weight Slider - only show when VLM is enabled */}
                  {vlmEnabled && vlmMask && (
                    <div>
                      <label style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: 10,
                        fontSize: '0.85rem',
                        letterSpacing: '0.08em',
                        color: 'rgba(160, 196, 255, 0.8)',
                      }}>
                        <span>AI Guidance Strength</span>
                        <span style={{ color: '#a0c4ff' }}>{Math.round(vlmWeight * 100)}%</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={vlmWeight * 100}
                        onChange={(e) => setVlmWeight(Number(e.target.value) / 100)}
                        className="control-slider"
                        style={{
                          width: '100%',
                          height: 20,
                          background: 'transparent',
                          cursor: 'pointer',
                          WebkitAppearance: 'none',
                        }}
                      />
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        fontSize: '0.7rem',
                        color: 'rgba(160, 196, 255, 0.4)',
                        marginTop: 5,
                      }}>
                        <span>Traditional</span>
                        <span>AI Enhanced</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Info Section */}
            <div style={{
              marginTop: 40,
              padding: '30px',
              background: 'rgba(20, 15, 30, 0.5)',
              borderRadius: 15,
              border: '1px solid rgba(212, 175, 55, 0.15)',
            }}>
              <h3 style={{
                fontFamily: '"Cinzel", serif',
                fontSize: '1.1rem',
                letterSpacing: '0.1em',
                color: '#d4af37',
                marginBottom: 20,
                textAlign: 'center',
              }}>
                ✧ Reading Tips ✧
              </h3>

              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: 20,
                fontSize: '0.9rem',
                color: 'rgba(232, 220, 200, 0.7)',
                lineHeight: 1.7,
              }}>
                <div>
                  <strong style={{ color: '#d4af37' }}>Sensitivity:</strong> Increase to reveal finer, more subtle lines. Decrease to focus only on the major palm lines.
                </div>
                <div>
                  <strong style={{ color: '#d4af37' }}>Thickness:</strong> Adjust to make detected lines more visible for easier reading.
                </div>
                <div>
                  <strong style={{ color: '#d4af37' }}>Best Results:</strong> Use a clear, high-contrast photo with good lighting and minimal shadows.
                </div>
                <div>
                  <strong style={{ color: '#a0c4ff' }}>AI Vision:</strong> Enable AI Vision mode for intelligent line detection. The AI identifies major palm lines (Heart, Head, Life, Fate) and enhances edge detection accuracy.
                </div>
              </div>
            </div>
          </>
        )}

        {/* Footer */}
        <footer style={{
          marginTop: 50,
          textAlign: 'center',
          fontSize: '0.8rem',
          color: 'rgba(232, 220, 200, 0.4)',
          letterSpacing: '0.05em',
        }}>
          <p>For entertainment and spiritual exploration purposes</p>
        </footer>
      </div>
    </div>
  );
}
