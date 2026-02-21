import type { Painting } from './types'

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
