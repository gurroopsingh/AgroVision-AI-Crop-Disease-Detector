# Task List: Full Hindi Support & Formatting Fixes

- [x] Update `translations.js` with Transparency page content (with HTML formatting).
- [x] Refactor `Transparency.jsx` to render translated HTML strings safely.
- [x] Create Python script to generate `disease_info_hi.json` with Hindi templates.
- [x] Run the Python script to generate `disease_info_hi.json`.
- [x] Update `app.py` to load `disease_info_hi.json` at startup.
- [x] Update `/predict` and `/disease-info` endpoints in `app.py` to accept `lang` parameter.
- [x] Update React frontend (`App.jsx`, `Home.jsx`, `Encyclopedia.jsx`) to pass `lang` to API requests.
- [x] Build and verify frontend changes.
