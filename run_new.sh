#!/usr/bin/env bash
# Launch the Nocturne redesign with its theme applied to THIS run only,
# so the original app.py keeps Streamlit's stock appearance.
exec streamlit run app_new.py \
  --theme.base dark \
  --theme.primaryColor "#9184d9" \
  --theme.backgroundColor "#161826" \
  --theme.secondaryBackgroundColor "#232532" \
  --theme.textColor "#e9e9ed" \
  --theme.font "sans serif" \
  --server.runOnSave true \
  "$@"
