import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const redis = getRedis()
  const raw = await redis.hgetall('stv:stats')

  if (!raw || !Object.keys(raw).length) {
    return res.status(503).json({ error: 'Stats not available' })
  }

  const stats = {
    total_listings: parseInt(raw.total_listings) || 0,
    regions: parseInt(raw.regions) || 0,
    price_min: parseFloat(raw.price_min) || 0,
    price_max: parseFloat(raw.price_max) || 0,
    price_median: parseFloat(raw.price_median) || 0,
    scored_count: parseInt(raw.scored_count) || 0,
    artists_count: parseInt(raw.artists_count) || 0,
    top_rated_count: parseInt(raw.top_rated_count) || 0,
    gems_count: parseInt(raw.gems_count) || 0,
    median_art_score: parseFloat(raw.median_art_score) || 0,
  }

  res.setHeader('Cache-Control', 's-maxage=60, stale-while-revalidate=300')
  return res.json(stats)
}
