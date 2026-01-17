// Vision Language Model Service for Palm Line Detection and Reading
// Uses Anthropic Claude for palm analysis

// Enhanced prompt for detailed palm line detection with control points
const PALM_ANALYSIS_PROMPT = `You are an expert palmistry analyst. Analyze this palm image carefully and identify ALL visible palm lines with precise coordinates.

For each line, provide:
1. Line name (Heart Line, Head Line, Life Line, Fate Line, Sun Line, Mercury Line, Marriage Lines, etc.)
2. Multiple control points along the line path (at least 4-6 points per line) as percentage coordinates (0-100 for both x and y, where 0,0 is top-left)
3. Line characteristics (depth: deep/medium/faint, length: long/medium/short, curvature: straight/slightly-curved/curved/very-curved)
4. Any special features (breaks, branches, islands, chains, crosses)

IMPORTANT COORDINATE GUIDELINES:
- Heart Line: Usually runs horizontally across upper palm, from edge near pinky towards index finger area (y typically 20-35%)
- Head Line: Below heart line, runs across middle of palm (y typically 40-55%)
- Life Line: Curves around the thumb base, starts between thumb and index finger (starts around x:30-40%, y:15-25%, curves down to x:20-35%, y:80-90%)
- Fate Line: Vertical line running up the center of palm (x typically 45-55%)
- Sun Line: Vertical line below ring finger area (x typically 60-70%)

Respond ONLY with valid JSON in this exact format:
{
  "lines": [
    {
      "name": "Heart Line",
      "points": [
        {"x": 85, "y": 28},
        {"x": 70, "y": 25},
        {"x": 55, "y": 23},
        {"x": 40, "y": 25},
        {"x": 25, "y": 30}
      ],
      "characteristics": {
        "depth": "deep",
        "length": "long",
        "curvature": "curved"
      },
      "features": ["slightly branched at end"]
    }
  ],
  "palmShape": "square|rectangular|conic|spatulate|mixed",
  "dominantElement": "earth|air|fire|water",
  "imageQuality": "excellent|good|moderate|poor",
  "overallImpression": "Brief description of the palm's notable features"
}`;

// Prompt for generating palm reading interpretation
const PALM_READING_PROMPT = `You are a mystical palm reader providing an insightful and personalized reading. Based on the palm analysis data provided, generate an engaging, detailed palm reading.

Palm Analysis Data:
{ANALYSIS_DATA}

Generate a palm reading that includes:
1. **Opening** - A mystical greeting acknowledging the unique nature of this palm
2. **The Heart Line** - Interpretation of emotional life, relationships, and love
3. **The Head Line** - Interpretation of intellect, thinking style, and decision making
4. **The Life Line** - Interpretation of vitality, life journey, and major life changes (NOT lifespan prediction)
5. **Other Lines** - Interpret any additional lines found (Fate, Sun, Mercury, etc.)
6. **Special Features** - Meaning of any notable features like branches, islands, or crosses
7. **Overall Reading** - A synthesized interpretation bringing all elements together
8. **Guidance** - Positive, empowering advice based on the reading

IMPORTANT GUIDELINES:
- Be mystical and engaging but respectful
- Focus on personality traits, tendencies, and potentials
- NEVER predict death, serious illness, or negative life events
- Keep the tone positive and empowering
- Use evocative, poetic language
- Make it personal and specific based on the actual line characteristics

Respond in JSON format:
{
  "opening": "Your mystical greeting...",
  "heartLine": {
    "title": "The Path of the Heart",
    "interpretation": "Detailed interpretation..."
  },
  "headLine": {
    "title": "The River of Thought",
    "interpretation": "Detailed interpretation..."
  },
  "lifeLine": {
    "title": "The Arc of Vitality",
    "interpretation": "Detailed interpretation..."
  },
  "otherLines": [
    {
      "name": "Fate Line",
      "title": "The Thread of Destiny",
      "interpretation": "Interpretation..."
    }
  ],
  "specialFeatures": "Interpretation of special features...",
  "overallReading": "Synthesized interpretation...",
  "guidance": "Empowering advice..."
}`;

// Convert image data URL to base64
const extractBase64FromDataUrl = (dataUrl) => {
  const base64Match = dataUrl.match(/^data:image\/\w+;base64,(.+)$/);
  return base64Match ? base64Match[1] : null;
};

// Anthropic Claude Vision API call
const analyzeWithAnthropic = async (imageDataUrl, apiKey, prompt) => {
  const base64Image = extractBase64FromDataUrl(imageDataUrl);
  if (!base64Image) {
    throw new Error('Invalid image data URL');
  }

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
      max_tokens: 3000,
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
              text: prompt,
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
  return data.content?.[0]?.text;
};

// Anthropic text completion for palm reading
const generateReadingWithAnthropic = async (analysisData, apiKey) => {
  const prompt = PALM_READING_PROMPT.replace('{ANALYSIS_DATA}', JSON.stringify(analysisData, null, 2));

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
      max_tokens: 2500,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error?.message || `Anthropic API error: ${response.status}`);
  }

  const data = await response.json();
  return data.content?.[0]?.text;
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
      palmShape: 'unknown',
      imageQuality: 'unclear',
      rawResponse: content,
    };
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]);
    return {
      lines: parsed.lines || [],
      palmShape: parsed.palmShape || 'unknown',
      dominantElement: parsed.dominantElement || 'unknown',
      imageQuality: parsed.imageQuality || 'moderate',
      overallImpression: parsed.overallImpression || '',
      rawResponse: content,
    };
  } catch (e) {
    console.warn('JSON parse error:', e);
    return {
      lines: [],
      palmShape: 'unknown',
      imageQuality: 'unclear',
      rawResponse: content,
    };
  }
};

// Parse reading response
const parseReadingResponse = (content) => {
  if (!content) {
    throw new Error('Empty response from LLM');
  }

  const jsonMatch = content.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    // Return raw content as overall reading if no JSON
    return {
      opening: 'The lines of your palm reveal a unique story...',
      overallReading: content,
      guidance: 'Trust in your journey and the wisdom your hands reveal.',
    };
  }

  try {
    return JSON.parse(jsonMatch[0]);
  } catch (e) {
    return {
      opening: 'The lines of your palm reveal a unique story...',
      overallReading: content,
      guidance: 'Trust in your journey and the wisdom your hands reveal.',
    };
  }
};

// Main analysis function
export const analyzeWithVLM = async (imageDataUrl, apiKey) => {
  if (!apiKey) {
    throw new Error('API key is required');
  }

  const content = await analyzeWithAnthropic(imageDataUrl, apiKey, PALM_ANALYSIS_PROMPT);
  return parseVLMResponse(content);
};

// Generate palm reading
export const generatePalmReading = async (analysisData, apiKey) => {
  if (!apiKey) {
    throw new Error('API key is required');
  }

  const content = await generateReadingWithAnthropic(analysisData, apiKey);
  return parseReadingResponse(content);
};

// Generate VLM-guided mask from line points
export const generateVLMGuidedMask = (vlmResult, width, height, lineWidth = 10) => {
  const mask = new Uint8Array(width * height);

  if (!vlmResult?.lines?.length) {
    return mask;
  }

  // Draw each line on the mask
  vlmResult.lines.forEach(line => {
    if (!line.points || line.points.length < 2) return;

    // Convert percentage points to pixel coordinates
    const pixelPoints = line.points.map(p => ({
      x: Math.round((p.x / 100) * width),
      y: Math.round((p.y / 100) * height),
    }));

    // Draw smooth curve through points using Catmull-Rom interpolation
    for (let i = 0; i < pixelPoints.length - 1; i++) {
      const p0 = pixelPoints[Math.max(0, i - 1)];
      const p1 = pixelPoints[i];
      const p2 = pixelPoints[i + 1];
      const p3 = pixelPoints[Math.min(pixelPoints.length - 1, i + 2)];

      // Interpolate between p1 and p2
      const steps = Math.max(
        Math.abs(p2.x - p1.x),
        Math.abs(p2.y - p1.y),
        20
      );

      for (let t = 0; t <= steps; t++) {
        const s = t / steps;
        const s2 = s * s;
        const s3 = s2 * s;

        // Catmull-Rom spline
        const x = Math.round(
          0.5 * (
            2 * p1.x +
            (-p0.x + p2.x) * s +
            (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * s2 +
            (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * s3
          )
        );
        const y = Math.round(
          0.5 * (
            2 * p1.y +
            (-p0.y + p2.y) * s +
            (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * s2 +
            (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * s3
          )
        );

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
  });

  return mask;
};

// Line colors for different palm lines
export const LINE_COLORS = {
  'Heart Line': { r: 255, g: 80, b: 120, hex: '#ff5078' },
  'Head Line': { r: 80, g: 180, b: 255, hex: '#50b4ff' },
  'Life Line': { r: 80, g: 220, b: 120, hex: '#50dc78' },
  'Fate Line': { r: 180, g: 100, b: 255, hex: '#b464ff' },
  'Sun Line': { r: 255, g: 200, b: 50, hex: '#ffc832' },
  'Mercury Line': { r: 255, g: 150, b: 50, hex: '#ff9632' },
  'Marriage Line': { r: 255, g: 130, b: 180, hex: '#ff82b4' },
  'Marriage Lines': { r: 255, g: 130, b: 180, hex: '#ff82b4' },
  'default': { r: 255, g: 200, b: 100, hex: '#ffc864' },
};

// Get color for a line name
export const getLineColor = (lineName) => {
  // Check for exact match first
  if (LINE_COLORS[lineName]) {
    return LINE_COLORS[lineName];
  }

  // Check for partial match
  for (const key of Object.keys(LINE_COLORS)) {
    if (lineName.toLowerCase().includes(key.toLowerCase().replace(' line', ''))) {
      return LINE_COLORS[key];
    }
  }

  return LINE_COLORS.default;
};
