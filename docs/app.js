(() => {
  "use strict";

  const config = window.EIP_CONFIG || {};
  const root = document.documentElement;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const desktopStory = window.matchMedia("(min-width: 900px) and (min-height: 700px)");
  const clamp = (n, min = 0, max = 1) => Math.min(max, Math.max(min, n));
  const smoothstep = (a, b, n) => { const t = clamp((n - a) / (b - a)); return t * t * (3 - 2 * t); };
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  // Empty publication URLs stay visibly unavailable, without dead # links.
  document.querySelectorAll("[data-link]").forEach(link => {
    const url = config.links?.[link.dataset.link];
    if (url) {
      link.href = url;
      link.hidden = false;
    } else if (link.dataset.link === "arxiv") {
      link.hidden = true;
    }
  });
  document.getElementById("arxiv-pending").hidden = Boolean(config.links?.arxiv);
  if (config.bibtex) {
    document.getElementById("bibtex-code").textContent = config.bibtex.trim();
    document.getElementById("citation-draft").hidden = true;
    document.getElementById("citation-note").textContent = "If you find this work useful, please consider citing our paper.";
  }

  const media = [];
  let heroPaused = reduceMotion.matches;
  const heroButton = document.getElementById("hero-toggle");
  const heroStatus = document.getElementById("hero-status");
  const mediaState = new WeakMap();

  function pauseAutomatically(video) {
    if (video.paused) return;
    mediaState.get(video).autoPausing = true;
    video.pause();
  }

  function syncVideo(video) {
    const state = mediaState.get(video);
    if (state.failed || !video.getAttribute("src")) return;
    const shouldPlay = state.visible && state.carouselActive && !document.hidden && !state.userPaused
      && (state.hero ? !heroPaused : (!reduceMotion.matches || state.userStarted));
    if (!shouldPlay) {
      pauseAutomatically(video);
    } else if (video.paused) {
      state.autoStarting = true;
      const attempt = video.play();
      if (attempt) attempt.catch(error => {
        state.autoStarting = false;
        // Native demo controls / the hero button remain available if autoplay is denied.
        if (state.hero && error.name === "NotAllowedError") {
          heroPaused = true;
          media.filter(v => mediaState.get(v).hero).forEach(pauseAutomatically);
          updateHeroUI();
        }
      });
    }
  }

  const loadObserver = new IntersectionObserver(entries => {
    entries.forEach(({ target, isIntersecting }) => {
      if (!isIntersecting) return;
      target.src = target.dataset.src;
      target.load();
      loadObserver.unobserve(target);
    });
  }, { rootMargin: "300px 0px" });

  const playbackObserver = new IntersectionObserver(entries => {
    entries.forEach(({ target, isIntersecting, intersectionRatio }) => {
      mediaState.get(target).visible = isIntersecting && intersectionRatio >= .15;
      syncVideo(target);
    });
  }, { threshold: [0, .15, .5] });

  function registerVideo(src, poster, label, hero, onError) {
    const video = el("video");
    video.muted = true;
    video.defaultMuted = true;
    video.setAttribute("muted", "");
    video.loop = true;
    video.playsInline = true;
    video.setAttribute("playsinline", "");
    video.preload = "none";
    video.controls = !hero;
    video.setAttribute("aria-label", label);
    if (poster) video.poster = poster;
    video.dataset.src = src;
    const state = { hero, visible: false, carouselActive: true, failed: false, userPaused: false, userStarted: false, autoPausing: false, autoStarting: false };
    mediaState.set(video, state);
    media.push(video);
    video.addEventListener("canplay", () => syncVideo(video));
    video.addEventListener("play", () => {
      state.userPaused = false;
      if (!state.autoStarting) state.userStarted = true;
      state.autoStarting = false;
    });
    video.addEventListener("pause", () => {
      if (state.autoPausing) state.autoPausing = false;
      else state.userPaused = true;
    });
    video.addEventListener("volumechange", () => {
      if (!video.muted && video.volume > 0) media.filter(other => other !== video).forEach(other => { other.muted = true; });
    });
    video.addEventListener("error", () => {
      state.failed = true;
      video.hidden = true;
      loadObserver.unobserve(video);
      playbackObserver.unobserve(video);
      onError?.();
      updateHeroUI();
    }, { once: true });
    loadObserver.observe(video);
    playbackObserver.observe(video);
    return video;
  }

  function updateHeroUI() {
    const count = media.filter(v => mediaState.get(v).hero && !mediaState.get(v).failed).length;
    heroButton.hidden = !count;
    heroButton.textContent = heroPaused ? "Play videos ▷" : "Pause videos Ⅱ";
    heroButton.setAttribute("aria-label", heroPaused ? "Play background videos" : "Pause background videos");
    heroStatus.textContent = count === 4 ? "Real-robot interactions · Muted loops"
      : count ? `${count} video${count > 1 ? "s" : ""} · Remaining slots show preview stills`
      : "Preview stills · Videos unavailable";
  }

  (config.realDemos || []).slice(0, 4).forEach((item, i) => {
    const cell = document.querySelector(`[data-hero="${i}"]`);
    const img = cell.querySelector("img");
    if (item.poster) img.src = item.poster;
    img.alt = `${item.title} — preview still`;
    const src = item.heroSrc || item.src;
    if (src) cell.append(registerVideo(src, item.poster, item.title, true));
  });

  heroButton.addEventListener("click", () => {
    heroPaused = !heroPaused;
    media.filter(v => mediaState.get(v).hero).forEach(video => {
      mediaState.get(video).userPaused = false;
      syncVideo(video);
    });
    updateHeroUI();
  });

  function makeDemo(item, i, simulation) {
    const article = el("article", "demo-card reveal");
    const frame = el("div", `demo-media${simulation ? " is-simulation" : ""}${!item.poster ? " simulation-placeholder" : ""}`);
    if (item.poster) {
      const poster = el("img");
      poster.src = item.poster;
      poster.alt = `${item.title} — preview still`;
      poster.loading = "lazy";
      frame.append(poster);
    } else {
      const number = el("span", "simulation-id", "Simulation");
      number.setAttribute("aria-hidden", "true");
      frame.append(number);
    }
    const placeholder = (failed = false) => {
      frame.classList.add("is-placeholder");
      frame.append(el("span", "demo-badge", simulation ? "Simulation" : "Real robot"));
      const label = el("div", "demo-placeholder-label");
      const text = el("span", "", failed ? "Video unavailable" : "Video coming soon");
      text.append(el("small", "", failed ? "Please check back later." : item.posterFromPaper ? "Preview still · Figure 1" : ""));
      label.append(text);
      frame.append(label);
    };
    if (item.src) frame.append(registerVideo(item.src, item.poster, item.title, false, () => placeholder(true)));
    else placeholder();
    const caption = el("div", "demo-caption");
    const copy = el("div");
    copy.append(el("h4", "", item.title), el("p", "", item.description));
    caption.append(copy);
    article.append(frame, caption);
    return article;
  }

  function makeRealCarousel(items) {
    const carousel = el("section", "demo-carousel reveal");
    carousel.setAttribute("aria-roledescription", "carousel");
    carousel.setAttribute("aria-label", "Real-world interaction demos");
    const stage = el("div", "carousel-stage");
    stage.id = "real-carousel-stage";
    stage.tabIndex = 0;
    stage.setAttribute("aria-label", "Use left and right arrow keys to switch interaction videos.");
    const caption = el("div", "carousel-caption");
    const title = el("h4");
    const description = el("p");
    caption.append(title, description);
    const controls = el("div", "carousel-controls");
    const previous = el("button", "carousel-arrow", "←");
    const next = el("button", "carousel-arrow", "→");
    previous.type = next.type = "button";
    previous.setAttribute("aria-label", "Previous real-world demo");
    next.setAttribute("aria-label", "Next real-world demo");
    previous.setAttribute("aria-controls", stage.id);
    next.setAttribute("aria-controls", stage.id);
    const pagination = el("div", "carousel-pagination");
    pagination.setAttribute("role", "group");
    pagination.setAttribute("aria-label", "Choose an interaction video");
    const dots = items.map((item, i) => {
      const dot = el("button", "carousel-dot");
      dot.type = "button";
      dot.setAttribute("aria-label", `Show ${item.title}`);
      dot.setAttribute("aria-controls", stage.id);
      dot.addEventListener("click", () => selectSlide(i, true));
      pagination.append(dot);
      return dot;
    });
    controls.append(previous, pagination, next);
    const status = el("span", "sr-only");
    status.setAttribute("role", "status");
    status.setAttribute("aria-atomic", "true");
    const slides = items.map((item, i) => {
      const slide = makeDemo(item, i, false);
      slide.classList.remove("reveal");
      slide.classList.add("stacked-slide");
      slide.querySelector(".demo-caption").remove();
      slide.setAttribute("role", "group");
      slide.setAttribute("aria-roledescription", "slide");
      slide.setAttribute("aria-label", `${i + 1} of ${items.length}: ${item.title}`);
      const preview = el("button", "carousel-preview");
      preview.type = "button";
      preview.tabIndex = -1; // The dots provide keyboard access to every preview.
      preview.setAttribute("aria-label", `Show ${item.title}`);
      preview.addEventListener("click", () => {
        selectSlide(i, true);
        stage.focus({ preventScroll: true });
      });
      slide.append(preview);
      stage.append(slide);
      return slide;
    });
    carousel.append(stage, caption, controls, status);
    let active = 0;
    function selectSlide(index, announce = false) {
      active = (index + slides.length) % slides.length;
      slides.forEach((slide, i) => {
        const selected = i === active;
        const offset = (i - active + slides.length) % slides.length;
        if (!selected && slide.contains(document.activeElement)) stage.focus({ preventScroll: true });
        slide.classList.toggle("is-active", selected);
        slide.classList.toggle("is-next", offset === 1);
        slide.classList.toggle("is-previous", offset === slides.length - 1);
        slide.setAttribute("aria-hidden", String(!selected));
        slide.querySelector(".carousel-preview").hidden = selected;
        dots[i].setAttribute("aria-pressed", String(selected));
        const video = slide.querySelector("video");
        if (video) {
          video.inert = !selected;
          video.controls = selected;
          mediaState.get(video).carouselActive = selected;
          syncVideo(video);
        }
      });
      title.textContent = items[active].title;
      description.textContent = items[active].description;
      if (announce) status.textContent = `Showing ${items[active].title}, ${active + 1} of ${slides.length}.`;
    }
    previous.addEventListener("click", () => selectSlide(active - 1, true));
    next.addEventListener("click", () => selectSlide(active + 1, true));
    stage.addEventListener("keydown", event => {
      // Keep native video control shortcuts intact when a video has focus.
      if (event.target !== stage) return;
      const target = { ArrowLeft: active - 1, ArrowRight: active + 1, Home: 0, End: slides.length - 1 }[event.key];
      if (target === undefined) return;
      event.preventDefault();
      selectSlide(target, true);
    });

    // Horizontal swipes / drags switch cards; vertical gestures still scroll the page.
    let drag = null;
    let blockClickUntil = 0;
    stage.addEventListener("pointerdown", event => {
      if (!event.isPrimary || event.button !== 0) return;
      const video = event.target.closest("video");
      if (video && event.clientY > video.getBoundingClientRect().bottom - 48) return;
      drag = { id: event.pointerId, x: event.clientX, y: event.clientY, moving: false };
    });
    stage.addEventListener("pointermove", event => {
      if (!drag || event.pointerId !== drag.id) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (!drag.moving && Math.abs(dx) > 10 && Math.abs(dx) > Math.abs(dy)) {
        drag.moving = true;
        stage.setPointerCapture(event.pointerId);
        stage.classList.add("is-dragging");
      }
      if (drag.moving) {
        event.preventDefault();
        stage.style.setProperty("--drag-offset", `${clamp(dx, -90, 90)}px`);
      }
    });
    function endDrag(event) {
      if (!drag || event.pointerId !== drag.id) return;
      const dx = event.clientX - drag.x;
      if (drag.moving) {
        blockClickUntil = performance.now() + 400;
        if (event.type === "pointerup" && Math.abs(dx) > 36) selectSlide(active - Math.sign(dx), true);
      }
      if (stage.hasPointerCapture(event.pointerId)) stage.releasePointerCapture(event.pointerId);
      drag = null;
      stage.classList.remove("is-dragging");
      stage.style.removeProperty("--drag-offset");
    }
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);
    stage.addEventListener("lostpointercapture", event => {
      // A touch initially captured by <video> bubbles this event when the stage takes over.
      if (event.target === stage) endDrag(event);
    });
    stage.addEventListener("click", event => {
      if (performance.now() < blockClickUntil) {
        event.preventDefault();
        event.stopImmediatePropagation();
      }
    }, true);
    stage.addEventListener("dragstart", event => event.preventDefault());
    let lastWheel = 0;
    let wheelDelta = 0;
    let wheelHandled = false;
    stage.addEventListener("wheel", event => {
      if (Math.abs(event.deltaX) <= Math.abs(event.deltaY)) return;
      event.preventDefault();
      const now = performance.now();
      if (now - lastWheel > 180) { wheelDelta = 0; wheelHandled = false; }
      lastWheel = now;
      wheelDelta += event.deltaX;
      if (!wheelHandled && Math.abs(wheelDelta) > 30) {
        selectSlide(active + Math.sign(wheelDelta), true);
        wheelHandled = true;
      }
    }, { passive: false });
    selectSlide(0);
    return carousel;
  }

  const realDemos = (config.realDemos || []).slice(0, 4);
  const realGrid = document.getElementById("real-demos");
  if (realDemos.length >= 2) {
    realGrid.append(makeRealCarousel(realDemos));
  } else {
    realDemos.forEach((item, i) => realGrid.append(makeDemo(item, i, false)));
  }
  (config.simulationDemos || []).slice(0, 2).forEach((item, i) => document.getElementById("sim-demos").append(makeDemo(item, i, true)));
  updateHeroUI();
  document.addEventListener("visibilitychange", () => media.forEach(syncVideo));

  // Method tabs own their timer and image state; video playback stays independent.
  function setupMethodTabs() {
    const method = document.getElementById("method");
    const region = document.getElementById("method-interactive");
    if (!method || !region) return;
    const tablist = region.querySelector('[role="tablist"]');
    const tabs = [...region.querySelectorAll('.contribution-tab')];
    const panels = tabs.map(tab => document.getElementById(tab.getAttribute("aria-controls")));
    const rotation = document.getElementById("method-rotation");
    const status = document.getElementById("method-status");
    const narrow = window.matchMedia("(max-width: 760px)");
    const imageLoads = new Map();
    const delay = 5000;
    let active = 0;
    let timer = null;
    let visible = false;
    let hovered = false;
    let keyboardFocused = false;
    let userPaused = false;
    let pageAway = false;
    let pending = false;
    let request = 0;

    function loadImage(index) {
      if (!imageLoads.has(index)) {
        const img = panels[index].querySelector("img");
        img.loading = "eager";
        // Reuse the same decoded <img> nodes for every visit to a tab.
        imageLoads.set(index, img.decode().then(() => true).catch(() => false));
      }
      return imageLoads.get(index);
    }

    function canRotate() {
      return visible && !hovered && !keyboardFocused && !userPaused
        && !document.hidden && !pageAway && !reduceMotion.matches;
    }

    function stopTimer() {
      clearTimeout(timer);
      timer = null;
    }

    function restartTimer() {
      stopTimer();
      if (canRotate() && !pending) {
        timer = setTimeout(() => {
          timer = null;
          select((active + 1) % tabs.length);
        }, delay);
      }
    }

    async function select(index, manual = false) {
      stopTimer();
      const currentRequest = ++request;
      if (index === active) {
        pending = false;
        restartTimer();
        return;
      }
      pending = true;
      const loaded = await loadImage(index);
      // Rapid clicks can resolve out of order on slow connections.
      if (currentRequest !== request) return;
      pending = false;
      if (loaded && (manual || canRotate())) {
        active = index;
        tabs.forEach((tab, i) => {
          const selected = i === active;
          tab.setAttribute("aria-selected", String(selected));
          tab.tabIndex = selected ? 0 : -1;
          panels[i].classList.toggle("is-active", selected);
          panels[i].setAttribute("aria-hidden", String(!selected));
          panels[i].inert = !selected;
        });
        status.textContent = "";
      } else if (!loaded && manual) {
        status.textContent = "This preview could not be loaded. Please refresh the page to try again.";
      }
      restartTimer();
    }

    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => select(index, true));
      tab.addEventListener("keydown", event => {
        let next;
        if (event.key === "Home") next = 0;
        else if (event.key === "End") next = tabs.length - 1;
        else if (event.key === "ArrowRight" || (narrow.matches && event.key === "ArrowDown")) next = (index + 1) % tabs.length;
        else if (event.key === "ArrowLeft" || (narrow.matches && event.key === "ArrowUp")) next = (index + tabs.length - 1) % tabs.length;
        if (next === undefined) return;
        event.preventDefault();
        tabs[next].focus({ preventScroll: true });
        select(next, true);
      });
    });
    function setOrientation() {
      tablist.setAttribute("aria-orientation", narrow.matches ? "vertical" : "horizontal");
    }
    setOrientation();
    narrow.addEventListener("change", setOrientation);

    region.addEventListener("pointerenter", event => {
      if (event.pointerType === "touch") return;
      hovered = true;
      restartTimer();
    });
    region.addEventListener("pointerleave", event => {
      if (event.pointerType === "touch") return;
      hovered = false;
      restartTimer();
    });
    function updateFocus() {
      const focused = document.activeElement;
      const paused = focused !== rotation && region.contains(focused) && focused.matches(":focus-visible");
      if (paused !== keyboardFocused) {
        keyboardFocused = paused;
        restartTimer();
      }
    }
    region.addEventListener("focusin", updateFocus);
    region.addEventListener("focusout", () => queueMicrotask(updateFocus));

    function updateRotation() {
      rotation.hidden = reduceMotion.matches;
      rotation.textContent = userPaused ? "Resume slideshow" : "Pause slideshow";
      restartTimer();
    }
    rotation.addEventListener("click", () => {
      userPaused = !userPaused;
      updateRotation();
    });
    reduceMotion.addEventListener("change", updateRotation);
    document.addEventListener("visibilitychange", restartTimer);
    window.addEventListener("pagehide", () => { pageAway = true; stopTimer(); });
    window.addEventListener("pageshow", () => { pageAway = false; restartTimer(); });

    // Load the other previews shortly before Method enters view, rather than at page startup.
    const preloadObserver = new IntersectionObserver(entries => {
      if (!entries.some(entry => entry.isIntersecting)) return;
      panels.forEach((_, index) => loadImage(index));
      preloadObserver.disconnect();
    }, { rootMargin: "600px 0px" });
    preloadObserver.observe(method);
    const visibilityObserver = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      restartTimer();
    }, { rootMargin: "-64px 0px 0px 0px" });
    visibilityObserver.observe(method);
    updateRotation();
  }
  setupMethodTabs();

  // Content sections reveal once. The hero remains continuously tied to scroll position.
  const revealObserver = new IntersectionObserver(entries => {
    entries.forEach(({ target, isIntersecting }) => {
      if (isIntersecting) { target.classList.add("is-visible"); revealObserver.unobserve(target); }
    });
  }, { threshold: .08, rootMargin: "0px 0px -22px 0px" });
  document.querySelectorAll(".reveal").forEach(node => revealObserver.observe(node));
  root.classList.add("motion-ready");

  const hero = document.getElementById("top");
  const sectionNodes = [...document.querySelectorAll("[data-section]")];
  const navLinks = [...document.querySelectorAll(".site-header nav a")];
  const progress = document.querySelector(".reading-progress > span");
  const header = document.getElementById("site-header");
  let scrolly = false;
  let framePending = false;

  function renderScroll() {
    framePending = false;
    const vh = window.innerHeight;
    const heroRect = hero.getBoundingClientRect();
    header.classList.toggle("is-light", heroRect.bottom < header.offsetHeight + 40);
    progress.style.transform = `scaleX(${clamp(window.scrollY / Math.max(1, root.scrollHeight - vh))})`;
    if (scrolly) {
      const hp = clamp(-heroRect.top / Math.max(1, hero.offsetHeight - vh));
      hero.style.setProperty("--hero-scale", 1 - .065 * hp);
      hero.style.setProperty("--hero-radius", `${26 * hp}px`);
      hero.style.setProperty("--hero-y", `${-60 * hp}px`);
      hero.style.setProperty("--hero-opacity", 1 - .84 * smoothstep(.12, 1, hp));
    }
    let activeId = "top";
    sectionNodes.forEach(section => { if (section.getBoundingClientRect().top <= vh * .4) activeId = section.id; });
    navLinks.forEach(link => {
      if (link.hash === `#${activeId}`) link.setAttribute("aria-current", "location");
      else link.removeAttribute("aria-current");
    });
  }

  function queueScroll() {
    if (!framePending) { framePending = true; requestAnimationFrame(renderScroll); }
  }

  function configureMotion() {
    scrolly = desktopStory.matches && !reduceMotion.matches;
    root.classList.toggle("is-scrolly", scrolly);
    if (!scrolly) {
      hero.removeAttribute("style");
    }
    if (reduceMotion.matches) {
      heroPaused = true;
      media.forEach(video => { mediaState.get(video).userStarted = false; });
    }
    media.forEach(syncVideo);
    updateHeroUI();
    queueScroll();
  }

  window.addEventListener("scroll", queueScroll, { passive: true });
  window.addEventListener("resize", queueScroll, { passive: true });
  window.addEventListener("load", queueScroll);
  desktopStory.addEventListener("change", configureMotion);
  reduceMotion.addEventListener("change", configureMotion);
  configureMotion();

  const copyButton = document.getElementById("copy-bibtex");
  let copyTimer;
  copyButton.addEventListener("click", async () => {
    const text = document.getElementById("bibtex-code").textContent;
    let copied = false;
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        copied = true;
      }
    } catch { /* Try the local-file / non-HTTPS fallback below. */ }
    if (!copied) {
      const textarea = el("textarea", "sr-only");
      textarea.value = text;
      document.body.append(textarea);
      textarea.select();
      try { copied = document.execCommand("copy"); } catch { copied = false; }
      textarea.remove();
      copyButton.focus({ preventScroll: true });
    }
    if (!copied) {
      const range = document.createRange();
      range.selectNodeContents(document.getElementById("bibtex-code"));
      window.getSelection().removeAllRanges();
      window.getSelection().addRange(range);
    }
    copyButton.textContent = copied ? "Copied ✓" : "Select & copy manually";
    document.getElementById("copy-status").textContent = copied ? "BibTeX copied to clipboard." : "Citation selected. Use your browser’s copy command.";
    clearTimeout(copyTimer);
    copyTimer = setTimeout(() => { copyButton.textContent = "Copy BibTeX ⧉"; }, 2500);
  });
})();
