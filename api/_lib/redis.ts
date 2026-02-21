import Redis from 'ioredis'

let redis: Redis | null = null

export function getRedis(): Redis {
  if (!redis) {
    const url = process.env.stevenson_REDIS_URL
    if (!url) throw new Error('stevenson_REDIS_URL is not set')
    redis = new Redis(url, { maxRetriesPerRequest: 3 })
  }
  return redis
}
