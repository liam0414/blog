import React from "react";
import { DocsThemeConfig } from "nextra-theme-docs";

const config: DocsThemeConfig = {
  logo: <span>My Projects</span>,
  project: {
    link: "https://github.com/ligam0414/blog",
  },
  docsRepositoryBase: "https://github.com/ligam0414/blog",
  footer: {
    text: "Liam Chen",
  },
  sidebar: {
    defaultMenuCollapseLevel: 1,
  },
  toc: {
    title: "Table of Contents",
  },
  editLink: {
    component: null,
  },
  feedback: {
    content: null,
  },
  useNextSeoProps() {
    return {
      titleTemplate: "%s - ˗ˏˋ ★ ˎˊ˗",
    };
  },
  faviconGlyph: "🚀",
};

export default config;
