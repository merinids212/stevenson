export interface Painting {
  id: string
  title: string
  price: number | null
  url: string
  location: string
  latitude: number | null
  longitude: number | null
  images: string[]
  posted: string
  region: string
  state: string
  quality_score: number | null
  clip_styles: { style: string; confidence: number }[] | null
  uniqueness: number | null
  art_score: number | null
  artist: string | null
  artist_confidence: number | null
  value_score: number | null
  aesthetic_score: number | null
  topiq_score: number | null
  musiq_score: number | null
  aesthetic2: number | null
  subjects: { tag: string; confidence: number }[] | null
  moods: { tag: string; confidence: number }[] | null
  medium_tags: { tag: string; confidence: number }[] | null
  colors: { hex: string; rgb: number[]; pct: number }[] | null
  color_tags: string[] | null
}

export interface Stats {
  total_listings: number
  regions: number
  price_min: number
  price_max: number
  price_median: number
  scored_count: number
  artists_count: number
  top_rated_count: number
  gems_count: number
  median_art_score: number
}

export interface PaintingsResponse {
  paintings: Painting[]
  total: number
}
