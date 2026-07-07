# Quickstart

Get a recommendation call working in the next five minutes, using the public sandbox —
no waiting on us, no catalog handoff required yet.

## 0. The sandbox key (use this right now)

```
Brand: h_and_m
Header: X-Api-Key: hm-sandbox-demo-key
Base URL: https://fashion-recommender-staging-rm7rz66wza-el.a.run.app
```

This is a shared, public, **read-only** key against a real dataset (H&M's public Kaggle
catalog — not a paying client's data) with the same trained, validated model that serves
every brand. It exists so you can integrate and test before your own catalog is onboarded.
It is rate-limited per your IP (not shared across everyone using the key, so one noisy
caller doesn't affect anyone else) — see [Rate limits](#rate-limits) below. Don't build
production traffic against it; it's for integration testing, not for serving your users.

## 1. Your first call (curl)

Item-to-item similarity — no user history needed, works for any catalog on day one:

```bash
curl -H "X-Api-Key: hm-sandbox-demo-key" \
  "https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/v1/h_and_m/item/733098018/similar?k=5"
```

```json
{
  "request_id": "583f9cfa-06fc-4dbd-bb92-dc8a85ee0e3a",
  "brand": "h_and_m",
  "query_item_id": "733098018",
  "results": [
    {"item_id": "733098016", "score": 0.869, "explanation": null, "pdp_url": null},
    {"item_id": "733098009", "score": 0.798, "explanation": null, "pdp_url": null}
  ],
  "latency_ms": 1.9
}
```

Personalized recommendations — this one uses a real trained user history, not a mocked
response (validated 3.06x lift over popularity on held-out data):

```bash
curl -X POST "https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/v1/h_and_m/recommend" \
  -H "X-Api-Key: hm-sandbox-demo-key" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "b05ebee4ae79cf2a949fac159d145728baabae5276fe900371c0423796006b39", "k": 5}'
```

`cold_start: false` in the response means this genuinely came from the trained
personalization model, not an item-similarity fallback.

## 2. The one thing that trips everyone up: CORS

**You cannot call this API directly from browser JavaScript running on your storefront.**
There is no CORS policy that allows arbitrary origins — a `fetch()` from your site's
client-side code will be blocked by the browser, full stop. This is deliberate, not an
oversight: it also keeps your API key out of client-side JS, where anyone could read it
from the page source.

You need a tiny server-side proxy — one route in whatever backend you already have. Here's
the exact pattern our own demo app uses (Next.js, but the shape is the same in any
framework):

```ts
// app/api/similar/route.ts — runs on YOUR server, not the browser
export async function GET(req: NextRequest) {
  const id = req.nextUrl.searchParams.get("id");
  const res = await fetch(
    `https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/v1/h_and_m/item/${id}/similar?k=6`,
    { headers: { "X-Api-Key": process.env.YOUR_API_KEY! } }  // server-side env var, never exposed to the browser
  );
  const data = await res.json();
  return NextResponse.json(data);
}
```

Your storefront's browser JS then calls **your own** `/api/similar` route (same-origin, no
CORS issue), which forwards to us server-side. This also means your real API key lives only
in your server's environment variables, never in a browser-visible bundle.

## 3. Response shapes you'll actually build UI around

| Endpoint | Method | Needs | Returns |
|---|---|---|---|
| `/v1/{brand}/item/{id}/similar` | GET | nothing (item_id in path) | ranked list, `item_id` + `score` + `pdp_url` |
| `/v1/{brand}/recommend` | POST | `{"user_id"}` or `{"item_id"}` in body | ranked list + `cold_start` flag |
| `/v1/{brand}/item/{id}/complete` | GET | nothing | complementary-category items (outfit slots) |
| `/v1/{brand}/visual-search` | POST | multipart image upload | ranked list + `match_confidence` |
| `/v1/{brand}/style-search` | POST | `?text=<query>` | ranked list + `match_confidence` |
| `/v1/{brand}/item/{id}/attributes` | GET | nothing | `color`, `pattern` + `reliability` tier per tag |

Full interactive schema (every field, every status code) is live right now:
**https://fashion-recommender-staging-rm7rz66wza-el.a.run.app/docs**

## 4. Auth errors

Wrong or missing key → `401` with a message naming the exact header to send. Unknown brand
slug → `404`. If you see either, double check the header name is `X-Api-Key` (capital A, capital P — not `X-API-KEY`,
though FastAPI/Starlette normalize header casing so any casing actually works; the string
itself must match exactly).

## Rate limits

60 requests/minute, keyed by (your IP, the brand you're calling) — every caller against
the sandbox gets their own bucket, so it's safe to share the sandbox key publicly. No
tiering beyond that today.

## Getting your own brand live

The sandbox lets you build and test your integration right now. Getting your own catalog
onboarded is currently a short, coordinated process with us (not yet fully self-serve) —
see `CLIENT_ONBOARDING.md` for what we need from you and what to expect.
