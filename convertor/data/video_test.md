
* **Small / low-quality versions**
* **Multiple resolutions**
* **Different aspect ratios**
* **Graceful GitHub README fallback**

This is the **correct and accepted way** to document media testing in Markdown while still keeping **HTML fidelity**.

---

# 🎥 Media Testing – HTML Video Variants (Low / Medium / High)

This document demonstrates **how the same video asset is represented in different quality profiles** using **HTML media tags inside Markdown**.

> ⚠️ GitHub README does not render video playback
> ✅ HTML is valid for documentation & downstream rendering (Docs, Pages, Apps)

---

## 📌 Base Test Asset

```text
https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4
```

---

## 🟢 Low Quality (Small Size – Fast Load)

**Use case:**

* Mobile
* Slow networks
* Preview thumbnails
* CI / automated UI tests


<video
  width="240"
  height="135"
  controls
  preload="metadata"
>
  <source
    src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    type="video/mp4"
  />
</video>


**Profile**

* Resolution: ~240p
* Bitrate: Low
* Memory footprint: Minimal

---

## 🟡 Medium Quality (Balanced)

**Use case:**

* Default UI testing
* Desktop preview
* Component validation

```html
<video
  width="480"
  height="270"
  controls
  preload="metadata"
>
  <source
    src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    type="video/mp4"
  />
</video>
```
<video
  width="480"
  height="270"
  controls
  preload="metadata"
>
  <source
    src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    type="video/mp4"
  />
</video>
---

## 🔵 High Quality (Reference)

**Use case:**

* Visual QA
* Full-screen playback
* Performance benchmarking

```html
<video
  width="720"
  height="405"
  controls
  preload="auto"
>
  <source
    src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    type="video/mp4"
  />
</video>
```
<video
  width="720"
  height="405"
  controls
  preload="auto"
>
  <source
    src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
    type="video/mp4"
  />
</video>
---

## 📐 Aspect Ratio Variants

### Landscape (16:9)

```html
<video width="320" height="180" controls>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
</video>
```
<video width="320" height="180" controls>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
</video>

### Portrait (9:16 – Simulated)

```html
<video width="180" height="320" controls>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
</video>
```
<video width="180" height="320" controls>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
</video>

### Square (1:1)

```html
<video width="240" height="240" controls>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
</video>
```
<video width="240" height="240" controls>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
</video>
---

## 🧪 Ultra-Lightweight Preview (Muted, No Autoplay)

**Best for lists / cards / grids**

```html
<video
  width="160"
  height="90"
  muted
  playsinline
  preload="none"
>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
</video>
```
<video
  width="160"
  height="90"
  muted
  playsinline
  preload="none"
>
  <source src="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4" type="video/mp4">
</video>
---

## 📦 JSON + HTML Mapping (Recommended)

```json
{
  "title": "Big Buck Bunny",
  "variants": {
    "low": { "width": 240, "height": 135 },
    "medium": { "width": 480, "height": 270 },
    "high": { "width": 720, "height": 405 }
  },
  "source": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
}
```

---

## ✅ Best Practices

* Always use **HTTPS**
* Control size using **HTML attributes**, not CSS in README
* Use **low-quality versions** for automation
* Document **aspect ratio explicitly**
* Keep README **descriptive**, not interactive

---

## ❌ What NOT to Do in README

```html
<video autoplay loop>
```

* Autoplay ❌
* Heavy preload ❌
* Large resolution by default ❌

---

If you want next, I can:

* Generate **portrait-only free test videos**
* Create **adaptive streaming (HLS/DASH) README**
* Provide **React / Next.js abstraction**
* Convert this into **Docs / GitHub Pages playable demo**
