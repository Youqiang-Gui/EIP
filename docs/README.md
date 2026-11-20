# EIP project page

This directory is the GitHub Pages publishing source. It contains static
HTML, CSS, JavaScript, figures, video posters, and compressed demo videos.
No Node.js installation or build step is required to publish or view it.

## Preview

From the repository root:

```bash
python -m http.server 8001 --bind 127.0.0.1 --directory docs
```

Open `http://localhost:8001`; forward port 8001 when working over SSH.

## Edit

- `index.html`: paper title, authors, abstract, contributions, and citation template.
- `styles.css`: layout, typography, and responsive behavior.
- `app.js`: page interactions and video playback behavior.
- `site-config.js`: resource links, videos, posters, and the official BibTeX entry.

Keep asset paths relative, such as `assets/videos/web/crack.mp4`, so they
resolve under the `/EIP/` project URL. Keep `.nojekyll` in this directory.

## Update videos

The published page uses only the six MP4 playback copies in `assets/videos/web/`.
Original recordings are not included. To regenerate the copies, place the local
originals in `assets/videos/` with these filenames:

```text
crack.mp4
cough.mp4
emotion.mp4
state adaptation.mp4
sim-01.mov
sim-02.mov
```

With Python 3.11+ and FFmpeg (`ffmpeg` and `ffprobe`) installed, run from the
repository root:

```bash
python docs/scripts/prepare_videos.py
```

The helper updates playback copies, posters, and cache versions. Commit the
changed `web/`, `posters/`, `site-config.js`, and `index.html` files together.
Original recordings and the local build-state file are ignored by Git.

## Update figures

Replace the corresponding PDF in `assets/figures/`, then regenerate previews:

```bash
python -m pip install pymupdf
python docs/scripts/render_figures.py
```

The helper renders all four displayed figures: `teaser`, `pipeline`, `mask`,
and `data-process`. Commit the PDFs and generated PNGs together.

## Publish

Configure GitHub Pages with **Deploy from a branch → main → /docs**.
After a successful deployment, the project URL is
<https://youqiang-gui.github.io/EIP/>.
