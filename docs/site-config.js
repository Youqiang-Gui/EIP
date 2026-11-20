// Project links and demo media paths, relative to index.html.
window.EIP_CONFIG = {
  links: {
    code: "https://github.com/Youqiang-Gui/EIP",
    arxiv: "", // e.g. https://arxiv.org/abs/your-paper-id
  },
  realDemos: [
    {
      title: "Proactive response — cracking sound",
      description: "Responding proactively to a cracking sound during interaction.",
      src: "assets/videos/web/crack.mp4?v=cbd9eb98cad4",
      heroSrc: "", // Optional shorter hero clip. Falls back to src.
      poster: "assets/posters/crack-video.jpg?v=86b5a11c024a",
      posterFromPaper: false,
    },
    {
      title: "Proactive response — coughing",
      description: "Responding proactively when the user coughs.",
      src: "assets/videos/web/cough.mp4?v=9b584e8490b8",
      heroSrc: "",
      poster: "assets/posters/cough-video.jpg?v=24ec782277a1",
      posterFromPaper: false,
    },
    {
      title: "Emotion-aware interaction",
      description: "Gestures that follow the emotional context of the conversation.",
      src: "assets/videos/web/emotion.mp4?v=94d8e6e176cb",
      heroSrc: "",
      poster: "assets/posters/emotion-video.jpg?v=57af265d6dcc",
      posterFromPaper: false,
    },
    {
      title: "State adaptation, interruption & recovery",
      description: "Adapting motion across interaction states, yielding to user interruptions, and recovering within the conversation.",
      src: "assets/videos/web/state adaptation.mp4?v=e0b287490b5c",
      heroSrc: "",
      poster: "assets/posters/state-adaptation-video.jpg?v=9a9923bb2a75",
      posterFromPaper: false,
    },
  ],
  simulationDemos: [
    {
      title: "Emotion-aware motion",
      description: "Expressive motion conditioned on the emotional context of the response.",
      src: "assets/videos/web/sim-01.mp4?v=3511f0667d1b",
      poster: "assets/posters/sim-01-video.jpg?v=c05b56083b40",
    },
    {
      title: "Proactive response, state adaptation & interruption",
      description: "Proactive responses to events, motion adaptation across interaction states, and reactions to user interruptions.",
      src: "assets/videos/web/sim-02.mp4?v=b6964574ba8d",
      poster: "assets/posters/sim-02-video.jpg?v=d77b0d0d3a99",
    },
  ],
  // null displays the citation placeholder until publication.
  bibtex: null,
};
