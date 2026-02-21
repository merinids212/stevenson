import type { Painting } from './types'

// Fields to fetch via HMGET (excludes embedding blob)
export const PAINTING_FIELDS = [
  'title', 'price', 'url', 'location', 'latitude', 'longitude',
  'images', 'posted', 'region', 'state', 'quality_score',
  'clip_styles', 'uniqueness', 'art_score', 'artist',
  'artist_confidence', 'value_score', 'aesthetic_score',
] as const

export function hmgetToHash(values: (string | null)[]): Record<string, string> {
  const hash: Record<string, string> = {}
  PAINTING_FIELDS.forEach((field, i) => {
    if (values[i] != null) hash[field] = values[i]!
  })
  return hash
}

function num(v: string | undefined): number | null {
  if (!v || v === 'null' || v === 'None') return null
  const n = parseFloat(v)
  return isNaN(n) ? null : n
}

function str(v: string | undefined): string | null {
  if (!v || v === 'null' || v === 'None') return null
  return v
}

function jsonParse<T>(v: string | undefined): T | null {
  if (!v || v === 'null' || v === 'None') return null
  try {
    return JSON.parse(v)
  } catch {
    return null
  }
}

// Title patterns that indicate non-art listings (house painting services, paint equipment, etc.)
const SPAM_PATTERNS = [
  /\bi will paint\b(?!ing)/i,
  /paint\/?repaint/i,
  /interior\s*[&+]\s*exterior\s*paint/i,
  /exterior\s*[&+]\s*interior\s*paint/i,
  /interior\s+(?:or\s+)?exterior\s+paint/i,
  /exterior\s+(?:or\s+)?interior\s+paint/i,
  /free\s+estimates?\b.*paint/i,
  /paint.*free\s+estimates?/i,
  /cheaper\s+than\s+the\s+competition/i,
  /\bhouse\s+painter\b/i,
  /\bhouse\s+paint\b(?!ing)/i,
  /paint\s+sprayer/i,
  /paint\s+crew/i,
  /\bairless\s+paint/i,
  /will\s+paint\s+your\s+(?:house|home|room|wall|fence|deck)/i,
]

export function isSpamTitle(title: string): boolean {
  return SPAM_PATTERNS.some(p => p.test(title))
}

export function parsePainting(hash: Record<string, string>, id: string): Painting {
  return {
    id,
    title: hash.title || '',
    price: num(hash.price),
    url: hash.url || '',
    location: hash.location || '',
    latitude: num(hash.latitude),
    longitude: num(hash.longitude),
    images: jsonParse<string[]>(hash.images) || [],
    posted: hash.posted || '',
    region: hash.region || '',
    state: hash.state || '',
    quality_score: num(hash.quality_score),
    clip_styles: jsonParse(hash.clip_styles),
    uniqueness: num(hash.uniqueness),
    art_score: num(hash.art_score),
    artist: str(hash.artist),
    artist_confidence: num(hash.artist_confidence),
    value_score: num(hash.value_score),
    aesthetic_score: num(hash.aesthetic_score),
  }
}
