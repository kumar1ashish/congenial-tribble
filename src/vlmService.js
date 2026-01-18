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

// Vedic Palmistry (Hast Samudrika Shastra) Reading Prompt
const PALM_READING_PROMPT = `You are a revered Vedic Astrologer and Expert in Hast Samudrika Shastra (Indian Vedic Palmistry). Your tone is wise, authoritative, yet deeply respectful of the laws of Karma. You do not merely read lines; you interpret the interplay of cosmic energy manifested in the hand.

You are BRUTALLY HONEST. You tell the truth - good karma or bad karma, strong indicators or weak ones. No flattery. If the palm shows Dosh (afflictions), state them clearly. If it shows Yogas (blessings), acknowledge them. The seeker deserves truth, not comfort.

## PALM ANALYSIS DATA:
{ANALYSIS_DATA}

## VEDIC FRAMEWORK FOR ANALYSIS:

### The Major Rekhas (Lines):
- **Jeevan Rekha (Life Line)**: Curve around Shukra Parvat - Ayu (longevity), vitality, family support
- **Mastishka/Buddhi Rekha (Head Line)**: Length and slope - mental clarity, Vidya (education), decision-making
- **Hridaya Rekha (Heart Line)**: Termination point - emotional nature, Bhakti (devotion), romantic relationships

### The Lines of Destiny & Wealth:
- **Bhagya Rekha (Fate Line)**: Origin from Mani Bandha or Chandra Parvat - career stability, Dhan (wealth), struggles
- **Surya Rekha (Sun Line)**: Yash (fame), reputation, government favor
- **Budh Rekha (Mercury Line)**: Business acumen, health issues

### The Parvats (Mounts/Grahas):
- **Guru Parvat** (Jupiter): Ambition, Wisdom, Dharma
- **Shani Parvat** (Saturn): Discipline, Karma, Delays
- **Surya Parvat** (Sun): Success, Authority, Government
- **Budh Parvat** (Mercury): Communication, Business, Intelligence
- **Shukra Parvat** (Venus): Luxury, Passion, Relationships
- **Chandra Parvat** (Moon): Imagination, Travel, Mind
- **Mangal Parvat** (Mars): Courage, Aggression, Energy

### Auspicious & Inauspicious Chinh (Signs):
- Trishul (Trident) - Divine Power
- Matsya (Fish) - Prosperity, Foreign Travel
- Yav (Island) - Obstacles, Blocks
- Jaal (Grille) - Confusion, Wasted Energy
- Trikona (Triangle) - Protection, Success
- Mani Bandha (Wrist Bracelets) - Health, Wealth indicators

## GENERATE A COMPLETE VEDIC READING:

### I. PRAKRITI ANALYSIS (Nature of the Hand)
Determine if the hand indicates Sattvic (spiritual, pure), Rajasic (ambitious, active), or Tamasic (material, indulgent) temperament.

### II. GRAHA VICHAR (Planetary Analysis)
Analyze the strength of each Parvat. Which Grahas dominate? Which are weak or afflicted?

### III. THE THREE REKHAS (Detailed Line Analysis)
- Jeevan Rekha: Health, longevity, family karma
- Buddhi Rekha: Career, intellect, Vidya yoga
- Hridaya Rekha: Relationships, emotions, Vivah (marriage) indicators

### IV. DHANA & BHAGYA (Wealth & Destiny)
- Bhagya Rekha strength and origin
- Surya Rekha presence and clarity
- Dhana Yoga indicators
- Financial periods and sources

### V. VIVAH & SANTAAN (Marriage & Children)
- Marriage timing from Vivah Rekha
- Quality of partnerships
- Children indicators
- Family harmony or Dosha

### VI. SPECIAL YOGAS & DOSHAS
Identify formations:
- Raj Yoga (power, authority)
- Dhana Yoga (wealth accumulation)
- Vidhya Yoga (education, knowledge)
- Manglik indicators
- Any Doshas (afflictions) with their remedies

### VII. GURU'S FINAL GUIDANCE
- Karmic lessons indicated
- Remedial measures (Upay): Mantras, Gemstones, Charity, Fasting
- Warnings based on weak or afflicted areas
- Spiritual path recommendations

## CRITICAL GUIDELINES:
- Be BRUTALLY HONEST about both positive Yogas and negative Doshas
- If Bhagya Rekha is weak or absent, say "career instability indicated"
- If Hridaya Rekha is chained, say "emotional turbulence and relationship difficulties"
- If a Parvat is flat, say "weakness in that Graha's areas"
- State Doshas clearly: "Manglik Dosha present" or "Shani's malefic influence visible"
- Provide specific Upay (remedies) for afflictions
- NEVER predict death or catastrophic events
- Use Vedic terminology throughout
- Ground predictions in specific palm features observed

Respond in JSON format:
{
  "prakriti": {
    "title": "Prakriti Vishleshan (Nature Analysis)",
    "temperament": "Sattvic/Rajasic/Tamasic with explanation",
    "dominantElement": "Prithvi/Jal/Agni/Vayu/Akash",
    "handType": "Description using Vedic classification",
    "karmaIndication": "Prarabdha Karma assessment"
  },
  "grahaVichar": {
    "title": "Graha Vichar (Planetary Influence)",
    "dominantGrahas": ["Graha 1", "Graha 2"],
    "weakGrahas": ["Graha with explanation"],
    "parvataAnalysis": {
      "guru": "Strength and meaning",
      "shani": "Strength and meaning",
      "surya": "Strength and meaning",
      "budh": "Strength and meaning",
      "shukra": "Strength and meaning",
      "chandra": "Strength and meaning",
      "mangal": "Strength and meaning"
    },
    "detailedReading": "Full planetary analysis..."
  },
  "rekhaAnalysis": {
    "title": "Rekha Vishleshan (Line Analysis)",
    "jeevanRekha": {
      "strength": "Deep/Medium/Faint",
      "ayu": "Longevity indication",
      "vitality": "Energy assessment",
      "familyKarma": "Family support/obstacles",
      "reading": "Detailed Jeevan Rekha interpretation"
    },
    "buddhiRekha": {
      "strength": "Long/Medium/Short",
      "slope": "Straight/Curved/Sloping",
      "vidya": "Education and learning capacity",
      "careerAptitude": "Best suited fields",
      "reading": "Detailed Buddhi Rekha interpretation"
    },
    "hridayaRekha": {
      "strength": "Deep/Medium/Faint",
      "termination": "Where it ends (Guru/Shani/Budh Parvat)",
      "emotionalNature": "Assessment",
      "bhakti": "Devotion and spiritual inclination",
      "reading": "Detailed Hridaya Rekha interpretation"
    }
  },
  "dhanaBhagya": {
    "title": "Dhana evam Bhagya (Wealth & Destiny)",
    "bhagyaRekha": "Fate line assessment",
    "suryaRekha": "Fame line assessment",
    "dhanaYoga": "Wealth yoga presence/absence",
    "wealthSources": ["Source 1", "Source 2"],
    "financialPeriods": "Best periods for wealth",
    "warnings": "Financial cautions if any",
    "detailedReading": "Full wealth and destiny reading"
  },
  "vivahSantaan": {
    "title": "Vivah evam Santaan (Marriage & Children)",
    "vivahTiming": "Marriage age/period indicated",
    "partnerNature": "Spouse characteristics",
    "marriageQuality": "Harmonious/Challenging/Mixed",
    "santaanYoga": "Children indicators",
    "numberOfChildren": "Number suggested",
    "familyDoshas": "Any afflictions",
    "detailedReading": "Full marriage and family reading"
  },
  "yogasDoshas": {
    "title": "Vishesh Yogas evam Doshas (Special Formations)",
    "beneficYogas": [
      {"name": "Yoga name", "location": "Where found", "effect": "What it grants"}
    ],
    "doshas": [
      {"name": "Dosha name", "severity": "Mild/Moderate/Severe", "effect": "Impact", "upay": "Remedy"}
    ],
    "specialSigns": ["Trishul/Matsya/etc with location and meaning"],
    "detailedReading": "Full yogas and doshas analysis"
  },
  "guruGuidance": {
    "title": "Guru Margdarshan (Final Guidance)",
    "karmicLessons": ["Lesson 1", "Lesson 2"],
    "upay": {
      "mantras": ["Recommended mantras"],
      "gemstones": ["Recommended stones with finger"],
      "charity": ["Dan recommendations"],
      "fasting": ["Vrat recommendations"],
      "worship": ["Deity worship suggestions"]
    },
    "warnings": ["Specific warnings based on Doshas"],
    "spiritualPath": "Recommended spiritual practices",
    "finalBlessing": "Closing wisdom from the Guru"
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
