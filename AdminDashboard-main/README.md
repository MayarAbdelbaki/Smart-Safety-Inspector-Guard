# SSIG Admin Dashboard

React app for viewing **PPE detections** and **fall alerts** from the same Supabase project used by the [Hugging Face PPE app](../huggingface/). No credentials or private data are stored in the code; configuration is via environment variables only.

## Setup

1. **Install dependencies**

   ```bash
   npm install
   ```

2. **Configure Supabase**

   Copy `.env.example` to `.env` and set:

   - `REACT_APP_SUPABASE_URL` — your Supabase project URL
   - `REACT_APP_SUPABASE_ANON_KEY` — your Supabase anon (public) key

   Use the same Supabase project as the Hugging Face Space. Do not commit `.env` (it is in `.gitignore`).

3. **Run locally**

   ```bash
   npm start
   ```

   Opens at `http://localhost:3000`. Routes: `/` (welcome), `/login`, `/dashboard`, `/guard`, `/fall-detection`, `/financial-penalty`. Protected routes require Supabase Auth (configure users in Supabase Dashboard if you use login).

## Data sources

| View            | Supabase table(s)       | Written by              |
|-----------------|-------------------------|--------------------------|
| Dashboard       | `detections_log`        | Hugging Face app         |
| Fall Detection  | `fall_alerts`           | Hugging Face app         |
| Guard           | `detections_log` (filter: unknown person) | — |
| Financial Penalty | `financial_penalties`, `person_info` | Your backend (optional) |

## Deployment

For deployment (e.g. Netlify, Vercel), set `REACT_APP_SUPABASE_URL` and `REACT_APP_SUPABASE_ANON_KEY` in the host’s environment variables. See `NETLIFY_DEPLOY.md` for Netlify-specific steps.
