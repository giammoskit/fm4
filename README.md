# Optional data backup

This folder is empty by default. The app always tries to download the prices from the Yahoo Finance API first.

If you want a backup for days when Yahoo Finance does not respond, set `SAVE_SNAPSHOT = True` in the last code cell of the Colab notebook, run that cell, download `data/prices_snapshot.csv` from the Colab Files panel and upload it here. The app then uses this copy only when the live download fails, and it shows a warning when it does.
