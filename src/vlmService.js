// Vision Language Model Service for Palm Line Detection and Reading
// Uses Anthropic Claude for comprehensive palm analysis

// Comprehensive prompt for detailed palm analysis
const PALM_ANALYSIS_PROMPT = `You are a world-renowned palmistry expert with decades of experience. Analyze this palm image with extreme precision and detail.

## REQUIRED ANALYSIS

### 1. MAJOR LINES (with coordinates as percentage 0-100, where 0,0 is top-left)
For each line provide: name, control points (4-6 points), depth (deep/medium/faint), length (long/medium/short), curvature, and special features.

- **Heart Line**: Emotional nature, relationships, cardiac health indicators
- **Head Line**: Intelligence type, mental approach, decision-making style
- **Life Line**: Vitality, life energy, major life changes (NOT lifespan)
- **Fate Line**: Career path, destiny, life direction
- **Sun Line (Apollo)**: Success, fame, creativity, public recognition
- **Mercury Line (Health)**: Business acumen, communication, health indicators
- **Marriage Lines**: Significant relationships, commitment patterns

### 2. MOUNTS (evaluate prominence: prominent/normal/flat)
- **Mount of Jupiter** (below index finger): Ambition, leadership, ego
- **Mount of Saturn** (below middle finger): Wisdom, responsibility, career
- **Mount of Apollo** (below ring finger): Creativity, success, artistry
- **Mount of Mercury** (below pinky): Communication, business, intelligence
- **Mount of Venus** (thumb base): Love, passion, vitality, sensuality
- **Mount of Luna** (opposite thumb): Imagination, intuition, creativity
- **Mount of Mars** (2 areas): Courage, aggression, resilience

### 3. FINGER ANALYSIS
- Finger lengths relative to each other
- Finger shapes (pointed/square/spatulate/conic)
- Finger spacing (wide/normal/close)
- Thumb characteristics (flexible/stiff, length, angle)

### 4. HAND CHARACTERISTICS
- Hand shape: Earth (square palm, short fingers) / Air (square palm, long fingers) / Fire (rectangular palm, short fingers) / Water (rectangular palm, long fingers)
- Skin texture: Fine/Medium/Coarse
- Flexibility indicators
- Dominant hand assessment if visible

### 5. SPECIAL MARKINGS (note location and significance)
- Stars, crosses, triangles, squares, grilles
- Islands, chains, breaks in lines
- Branches (upward=positive, downward=challenges)
- Sister lines, influence lines

### 6. KEY INDICATORS TO ASSESS
Based on the palm features, evaluate indicators for:
- **Career/Occupation tendencies**: Creative, analytical, leadership, service, technical, entrepreneurial
- **Financial potential**: Strong/moderate/developing wealth indicators
- **Relationship patterns**: Romantic nature, marriage timing indicators, partnership style
- **Children indicators**: Lines and markings suggesting offspring
- **Success potential**: Fame lines, achievement markers
- **Personality type**: Introvert/extrovert, emotional/logical, creative/practical
- **Social standing indicators**: Leadership marks, public recognition signs

Respond ONLY with valid JSON:
{
  "lines": [
    {
      "name": "Heart Line",
      "points": [{"x": 85, "y": 28}, {"x": 70, "y": 25}, {"x": 55, "y": 23}, {"x": 40, "y": 25}],
      "characteristics": {"depth": "deep", "length": "long", "curvature": "curved"},
      "features": ["branches at end", "clear and unbroken"],
      "interpretation": "Strong emotional nature, deep capacity for love"
    }
  ],
  "mounts": {
    "jupiter": {"prominence": "prominent", "meaning": "Strong ambition and leadership"},
    "saturn": {"prominence": "normal", "meaning": "Balanced responsibility"},
    "apollo": {"prominence": "prominent", "meaning": "Creative success indicated"},
    "mercury": {"prominence": "normal", "meaning": "Good communication skills"},
    "venus": {"prominence": "prominent", "meaning": "Passionate nature, strong vitality"},
    "luna": {"prominence": "normal", "meaning": "Good imagination"},
    "mars": {"prominence": "normal", "meaning": "Adequate courage and resilience"}
  },
  "fingers": {
    "shapes": "Mixed conic and square",
    "spacing": "Normal spacing with slight gap between index and middle",
    "lengths": "Index finger nearly equal to ring finger",
    "thumb": "Strong, flexible thumb indicating adaptability"
  },
  "handShape": "Fire hand - rectangular palm with shorter fingers",
  "skinTexture": "Medium texture indicating balance of sensitivity and practicality",
  "specialMarkings": [
    {"type": "star", "location": "Mount of Apollo", "meaning": "Potential for recognition and success"},
    {"type": "triangle", "location": "Center of palm", "meaning": "Good fortune and intellectual achievement"}
  ],
  "keyIndicators": {
    "careerType": "Leadership and creative fields favored",
    "financialPotential": "Strong wealth accumulation potential",
    "relationshipStyle": "Passionate, committed, seeks deep connection",
    "childrenIndicators": "2-3 prominent lines suggesting children",
    "successPotential": "High - multiple success markers present",
    "personalityType": "Extroverted, emotionally intelligent, creative",
    "socialStanding": "Natural leadership, likely to achieve prominence"
  },
  "genderIndicators": "Appears to be [male/female] based on hand structure",
  "dominantElement": "Fire - passionate, energetic, action-oriented",
  "imageQuality": "good",
  "overallImpression": "A hand showing strong potential for success, deep emotional capacity, and natural leadership abilities"
}`;

// Comprehensive prompt for detailed palm reading generation
const PALM_READING_PROMPT = `You are a no-nonsense master palmist known for brutally honest readings. You tell people the truth - good or bad. No flattery, no sugarcoating. Based on the detailed palm analysis below, generate specific, HONEST predictions and insights. If the palm shows weakness, say it. If it shows strength, say it. Be direct.

## PALM ANALYSIS DATA:
{ANALYSIS_DATA}

## GENERATE A COMPLETE READING WITH THESE SECTIONS:

### 1. OPENING
A direct, no-nonsense introduction. Skip the mystical fluff. State what you see immediately.

### 2. PERSONALITY PROFILE
- Core personality traits
- Emotional nature (from Heart Line)
- Mental approach (from Head Line)
- Temperament (from hand shape and mounts)
- Strengths and growth areas

### 3. CAREER & OCCUPATION
Based on the lines, mounts, and finger characteristics, predict:
- Most suitable career paths (be specific: e.g., "medicine, particularly surgery" not just "healthcare")
- Leadership potential and style
- Entrepreneurial indicators
- Creative vs analytical career alignment
- Peak career periods

### 4. FINANCIAL OUTLOOK
- Wealth accumulation potential
- Money management style
- Best periods for financial growth
- Sources of income (salary, business, investments, inheritance)
- Financial advice based on palm indicators

### 5. MARRIAGE & RELATIONSHIPS
- Romantic nature and love style
- Ideal partner characteristics
- Marriage timing indicators (early/mid/later in life)
- Number of significant relationships indicated
- Relationship challenges and strengths
- Compatibility indicators

### 6. CHILDREN & FAMILY
- Children indicators (number suggested by lines)
- Parenting style indicated
- Family harmony indicators
- Relationship with children

### 7. SUCCESS & ACHIEVEMENT
- Fame and recognition potential
- Areas of greatest success
- Timing of major achievements
- Public vs private success orientation
- Legacy indicators

### 8. SOCIAL STATUS & PUBLIC LIFE
- Natural social position
- Leadership in community
- Public recognition potential
- Social circle characteristics
- Influence and impact on others

### 9. HEALTH & VITALITY
- Overall vitality indicators (from Life Line)
- Areas requiring attention
- Energy levels and stamina
- Longevity indicators (vitality, NOT death prediction)

### 10. LIFE PATH SUMMARY
A comprehensive synthesis bringing all elements together into a cohesive life narrative.

### 11. GUIDANCE & ADVICE
Specific, actionable advice based on the reading.

## IMPORTANT GUIDELINES:
- Be BRUTALLY HONEST - do not sugarcoat or flatter
- Tell it like it is: good, bad, ugly, whatever the palm shows
- Avoid sycophancy - if indicators are weak, say so directly
- Use the actual palm data to justify predictions
- If a line is faint or broken, state the negative implications clearly
- Mention genuine weaknesses, challenges, and areas of concern
- Give specific warnings where palm indicates difficulties
- Balance does NOT mean forced positivity - if the palm shows struggle, say it
- Include timing when indicators suggest it
- Be direct about limitations: "Your fate line is weak, suggesting career instability"
- Do NOT use phrases like "wonderful potential" or "blessed with" unless truly warranted
- NEVER predict death or catastrophic events
- Be a straight-talking advisor, not a flattering fortune teller

Respond in JSON format:
{
  "opening": "Personalized mystical greeting...",
  "personality": {
    "title": "Your Inner Nature",
    "coreTraits": ["trait1", "trait2", "trait3"],
    "emotionalNature": "Detailed description...",
    "mentalApproach": "Detailed description...",
    "temperament": "Detailed description...",
    "strengths": ["strength1", "strength2"],
    "growthAreas": ["area1", "area2"]
  },
  "career": {
    "title": "Your Professional Destiny",
    "suitablePaths": ["Specific career 1", "Specific career 2", "Specific career 3"],
    "leadershipStyle": "Description...",
    "entrepreneurialPotential": "High/Medium/Low with explanation",
    "careerOrientation": "Creative/Analytical/Leadership/Service",
    "peakPeriods": "Description of timing...",
    "detailedReading": "Full career interpretation..."
  },
  "finance": {
    "title": "Your Wealth Potential",
    "wealthPotential": "Strong/Moderate/Developing",
    "moneyStyle": "Description of financial behavior...",
    "incomeSources": ["Primary source", "Secondary source"],
    "bestPeriods": "Description...",
    "advice": "Specific financial guidance...",
    "detailedReading": "Full financial interpretation..."
  },
  "marriage": {
    "title": "Your Heart's Journey",
    "romanticNature": "Description...",
    "idealPartner": "Characteristics description...",
    "marriageTiming": "Specific timing indication...",
    "numberOfRelationships": "Indicated number...",
    "relationshipStrengths": ["strength1", "strength2"],
    "challenges": ["challenge1"],
    "detailedReading": "Full relationship interpretation..."
  },
  "children": {
    "title": "Your Legacy of Love",
    "childrenIndicated": "Number or range...",
    "parentingStyle": "Description...",
    "familyHarmony": "Description...",
    "detailedReading": "Full children/family interpretation..."
  },
  "success": {
    "title": "Your Path to Achievement",
    "famePotential": "High/Moderate/Private success orientation",
    "areasOfSuccess": ["Area 1", "Area 2"],
    "achievementTiming": "Description...",
    "legacyIndicators": "Description...",
    "detailedReading": "Full success interpretation..."
  },
  "socialStatus": {
    "title": "Your Place in the World",
    "naturalPosition": "Description...",
    "leadershipRole": "Description...",
    "publicRecognition": "Description...",
    "influence": "Description...",
    "detailedReading": "Full social status interpretation..."
  },
  "health": {
    "title": "Your Vitality",
    "overallVitality": "Strong/Good/Moderate",
    "energyLevels": "Description...",
    "areasOfAttention": ["Area 1"],
    "detailedReading": "Full health interpretation..."
  },
  "lifePath": {
    "title": "Your Life's Narrative",
    "summary": "Comprehensive life path synthesis..."
  },
  "guidance": {
    "title": "Wisdom for Your Journey",
    "advice": ["Specific advice 1", "Specific advice 2", "Specific advice 3"],
    "affirmation": "A powerful closing affirmation..."
  }
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
      max_tokens: 4096,
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
      max_tokens: 4096,
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
      mounts: parsed.mounts || {},
      fingers: parsed.fingers || {},
      handShape: parsed.handShape || 'unknown',
      skinTexture: parsed.skinTexture || 'unknown',
      specialMarkings: parsed.specialMarkings || [],
      keyIndicators: parsed.keyIndicators || {},
      genderIndicators: parsed.genderIndicators || 'unknown',
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
    return {
      opening: 'The lines of your palm reveal a unique story...',
      overallReading: content,
      guidance: { advice: ['Trust in your journey and the wisdom your hands reveal.'] },
    };
  }

  try {
    return JSON.parse(jsonMatch[0]);
  } catch (e) {
    return {
      opening: 'The lines of your palm reveal a unique story...',
      overallReading: content,
      guidance: { advice: ['Trust in your journey and the wisdom your hands reveal.'] },
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

      const steps = Math.max(
        Math.abs(p2.x - p1.x),
        Math.abs(p2.y - p1.y),
        20
      );

      for (let t = 0; t <= steps; t++) {
        const s = t / steps;
        const s2 = s * s;
        const s3 = s2 * s;

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
  if (LINE_COLORS[lineName]) {
    return LINE_COLORS[lineName];
  }

  for (const key of Object.keys(LINE_COLORS)) {
    if (lineName.toLowerCase().includes(key.toLowerCase().replace(' line', ''))) {
      return LINE_COLORS[key];
    }
  }

  return LINE_COLORS.default;
};
