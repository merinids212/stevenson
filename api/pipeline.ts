import type { VercelRequest, VercelResponse } from '@vercel/node'
import { getRedis } from './_lib/redis'

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const redis = getRedis()

  // Get overall stats
  const raw = await redis.hgetall('stv:stats')

  // Get per-source counts from sorted set indexes
  const clCount = await redis.zcard('stv:idx:source:cl')
  const ebCount = await redis.zcard('stv:idx:source:eb')
  const totalIndexed = await redis.zcard('stv:idx:art_score')
  const dedupCount = await redis.scard('stv:dedup')

  // Sample some scored paintings to get score range from Redis
  const topScored = await redis.zrevrange('stv:idx:art_score', 0, 0, 'WITHSCORES')
  const bottomScored = await redis.zrange('stv:idx:art_score', 0, 0, 'WITHSCORES')

  // Per-source scored counts (paintings with non-zero art_score)
  // Use ZRANGEBYSCORE to count paintings with score > 0
  const clScored = await redis.zcount('stv:idx:source:cl', 0.1, '+inf')
  const ebScored = await redis.zcount('stv:idx:source:eb', 0.1, '+inf')

  const pipeline = {
    total_listings: parseInt(raw.total_listings) || 0,
    scored_count: parseInt(raw.scored_count) || 0,
    unscored_count: (parseInt(raw.total_listings) || 0) - (parseInt(raw.scored_count) || 0),
    score_pct: parseInt(raw.total_listings) > 0
      ? Math.round((parseInt(raw.scored_count) || 0) / parseInt(raw.total_listings) * 100)
      : 0,
    sources: {
      craigslist: { total: clCount, scored: clScored, unscored: clCount - clScored },
      ebay: { total: ebCount, scored: ebScored, unscored: ebCount - ebScored },
    },
    dedup_count: dedupCount,
    indexed_count: totalIndexed,
    score_max: topScored.length >= 2 ? parseFloat(topScored[1]) : 0,
    score_min: bottomScored.length >= 2 ? parseFloat(bottomScored[1]) : 0,
    artists_count: parseInt(raw.artists_count) || 0,
    top_rated_count: parseInt(raw.top_rated_count) || 0,
    gems_count: parseInt(raw.gems_count) || 0,
    regions: parseInt(raw.regions) || 0,
    states: parseInt(raw.states) || 0,
    median_art_score: parseFloat(raw.median_art_score) || 0,
    price_min: parseFloat(raw.price_min) || 0,
    price_max: parseFloat(raw.price_max) || 0,
    price_median: parseFloat(raw.price_median) || 0,
  }

  res.setHeader('Cache-Control', 's-maxage=10, stale-while-revalidate=30')
  return res.json(pipeline)
}
