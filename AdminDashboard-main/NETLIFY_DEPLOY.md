# Netlify Deployment Guide

## Quick Setup

1. **Push your code to GitHub** (if not already done)

2. **Connect to Netlify:**
   - Go to [Netlify](https://app.netlify.com)
   - Click "Add new site" → "Import an existing project"
   - Connect to your GitHub repository
   - Select the repository containing the frontend folder

3. **Configure Build Settings:**
   - **Base directory:** `frontend` (if your repo root contains the frontend folder)
   - **Build command:** `npm run build` (or leave default)
   - **Publish directory:** `build` (this is set in `netlify.toml` automatically)

4. **Set Environment Variables:**
   - Go to Site settings → Environment variables
   - Add the following variables:
     - `REACT_APP_SUPABASE_URL` = Your Supabase project URL
     - `REACT_APP_SUPABASE_ANON_KEY` = Your Supabase anon key

5. **Deploy:**
   - Click "Deploy site"
   - Netlify will automatically build and deploy your site

## Important Notes

- The `netlify.toml` file is already configured to:
  - Build using `npm run build`
  - Publish from the `build` directory
  - Handle React Router redirects (all routes redirect to index.html)

- **Environment Variables:** Make sure to set `REACT_APP_SUPABASE_URL` and `REACT_APP_SUPABASE_ANON_KEY` in Netlify's environment variables section. These are required for the app to connect to Supabase.

- **Base Directory:** If your GitHub repo has the frontend folder at the root level, you may need to set the base directory to `frontend` in Netlify's build settings.

## Troubleshooting

- **Build fails:** Check that all dependencies are in `package.json`
- **Environment variables not working:** Make sure they start with `REACT_APP_` prefix
- **Routing issues:** The `netlify.toml` includes redirect rules for React Router
