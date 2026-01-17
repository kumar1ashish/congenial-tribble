// Vision Language Model Service for Palm Line Detection
// Supports multiple VLM providers for enhanced palm line analysis

const VLM_PROVIDERS = {
  OPENAI: 'openai',
  ANTHROPIC: 'anthropic',
};

// Prompt for palm line detection and analysis
const PALM_ANALYSIS_PROMPT = `Analyze this palm image and identify the major palm lines. For each line found, provide:
1. The line name (Heart Line, Head Line, Life Line, Fate Line, Sun Line, etc.)
2. Its approximate position described as coordinates relative to the palm (use percentages from 0-100 for x and y positions)
3. The line's characteristics (length, depth, curvature)

Respond in JSON format with this structure:
{
  "lines": [
    {
      "name": "Heart Line",
      "startX": 10,
      "startY": 30,
      "endX": 90,
      "endY": 25,
      "curvePoints": [{"x": 50, "y": 20}],
      "characteristics": "Deep, curved, extends across palm"
    }
  ],
  "palmQuality": "clear|moderate|unclear",
  "suggestions": "Any suggestions for better detection"
}

Focus on accuracy and provide coordinates that trace each line's path across the palm.`;

// Convert image data URL to base64
const extractBase64FromDataUrl = (dataUrl) => {
  const base64Match = dataUrl.match(/^data:image\/\w+;base64,(.+)$/);
  return base64Match ? base64Match[1] : null;
};

// OpenAI GPT-4 Vision API call
const analyzeWithOpenAI = async (imageDataUrl, apiKey) => {
  const base64Image = extractBase64FromDataUrl(imageDataUrl);
  if (!base64Image) {
    throw new Error('Invalid image data URL');
  }

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: PALM_ANALYSIS_PROMPT,
            },
            {
              type: 'image_url',
              image_url: {
                url: imageDataUrl,
                detail: 'high',
              },
            },
          ],
        },
      ],
      max_tokens: 2000,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `OpenAI API error: ${response.status}`);
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content;

  return parseVLMResponse(content);
};

// Anthropic Claude Vision API call
const analyzeWithAnthropic = async (imageDataUrl, apiKey) => {
  const base64Image = extractBase64FromDataUrl(imageDataUrl);
  if (!base64Image) {
    throw new Error('Invalid image data URL');
  }

  // Determine media type from data URL
  const mediaTypeMatch = imageDataUrl.match(/^data:(image\/\w+);base64,/);
  const mediaType = mediaTypeMatch ? mediaTypeMatch[1] : 'image/jpeg';

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2000,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: mediaType,
                data: base64Image,
              },
            },
            {
              type: 'text',
              text: PALM_ANALYSIS_PROMPT,
            },
          ],
        },
      ],
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Anthropic API error: ${response.status}`);
  }

  const data = await response.json();
  const content = data.content?.[0]?.text;

  return parseVLMResponse(content);
};

// Parse VLM response and extract JSON
const parseVLMResponse = (content) => {
  if (!content) {
    throw new Error('Empty response from VLM');
  }

  // Try to extract JSON from the response
  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    console.warn('Could not extract JSON from VLM response:', content);
    return {
      lines: [],
      palmQuality: 'unclear',
      suggestions: 'Could not parse VLM response',
      rawResponse: content,
    };
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]);
    return {
      lines: parsed.lines || [],
      palmQuality: parsed.palmQuality || 'moderate',
      suggestions: parsed.suggestions || '',
      rawResponse: content,
    };
  } catch (e) {
    console.warn('JSON parse error:', e);
    return {
      lines: [],
      palmQuality: 'unclear',
      suggestions: 'Could not parse VLM response',
      rawResponse: content,
    };
  }
};

// Main analysis function
export const analyzeWithVLM = async (imageDataUrl, provider, apiKey) => {
  if (!apiKey) {
    throw new Error('API key is required');
  }

  switch (provider) {
    case VLM_PROVIDERS.OPENAI:
      return analyzeWithOpenAI(imageDataUrl, apiKey);
    case VLM_PROVIDERS.ANTHROPIC:
      return analyzeWithAnthropic(imageDataUrl, apiKey);
    default:
      throw new Error(`Unknown VLM provider: ${provider}`);
  }
};

// Generate enhanced edge mask based on VLM-detected lines
export const generateVLMGuidedMask = (vlmResult, width, height, lineWidth = 8) => {
  const mask = new Uint8Array(width * height);

  if (!vlmResult?.lines?.length) {
    return mask;
  }

  // Helper to draw a thick line on the mask
  const drawLine = (x1, y1, x2, y2, curvePoints = []) => {
    // Convert percentage coordinates to pixel coordinates
    const px1 = Math.round((x1 / 100) * width);
    const py1 = Math.round((y1 / 100) * height);
    const px2 = Math.round((x2 / 100) * width);
    const py2 = Math.round((y2 / 100) * height);

    // Build list of points including curve points
    const points = [{ x: px1, y: py1 }];

    if (curvePoints && curvePoints.length > 0) {
      curvePoints.forEach(cp => {
        points.push({
          x: Math.round((cp.x / 100) * width),
          y: Math.round((cp.y / 100) * height),
        });
      });
    }

    points.push({ x: px2, y: py2 });

    // Draw Bezier-like curve through all points
    for (let i = 0; i < points.length - 1; i++) {
      const p1 = points[i];
      const p2 = points[i + 1];

      // Number of steps based on distance
      const dist = Math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2);
      const steps = Math.max(Math.round(dist), 1);

      for (let t = 0; t <= steps; t++) {
        const ratio = t / steps;
        const x = Math.round(p1.x + (p2.x - p1.x) * ratio);
        const y = Math.round(p1.y + (p2.y - p1.y) * ratio);

        // Draw thick point
        for (let dy = -lineWidth; dy <= lineWidth; dy++) {
          for (let dx = -lineWidth; dx <= lineWidth; dx++) {
            if (dx * dx + dy * dy <= lineWidth * lineWidth) {
              const nx = x + dx;
              const ny = y + dy;
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                mask[ny * width + nx] = 255;
              }
            }
          }
        }
      }
    }
  };

  // Draw all detected lines
  vlmResult.lines.forEach(line => {
    if (line.startX !== undefined && line.startY !== undefined &&
        line.endX !== undefined && line.endY !== undefined) {
      drawLine(line.startX, line.startY, line.endX, line.endY, line.curvePoints);
    }
  });

  return mask;
};

// Combine traditional edge detection with VLM-guided mask
export const combineEdgeDetectionWithVLM = (traditionalMask, vlmMask, width, height, vlmWeight = 0.5) => {
  const combined = new Uint8Array(width * height);

  for (let i = 0; i < width * height; i++) {
    // Boost edges where VLM detected lines
    if (vlmMask[i] > 0) {
      // Strongly enhance VLM-detected areas
      combined[i] = Math.min(255, traditionalMask[i] + vlmMask[i] * vlmWeight);
    } else {
      // Reduce non-VLM areas slightly to reduce noise
      combined[i] = Math.max(0, traditionalMask[i] * (1 - vlmWeight * 0.3));
    }
  }

  return combined;
};

// Line colors for different palm lines
export const LINE_COLORS = {
  'Heart Line': { r: 255, g: 100, b: 100 },      // Red
  'Head Line': { r: 100, g: 200, b: 255 },       // Blue
  'Life Line': { r: 100, g: 255, b: 100 },       // Green
  'Fate Line': { r: 200, g: 100, b: 255 },       // Purple
  'Sun Line': { r: 255, g: 200, b: 50 },         // Gold
  'Mercury Line': { r: 255, g: 150, b: 50 },     // Orange
  'Marriage Line': { r: 255, g: 150, b: 200 },   // Pink
  'default': { r: 255, g: 200, b: 100 },         // Default gold
};

export { VLM_PROVIDERS };
