import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Matches the screenshot's near-black background with a slightly
        // lighter surface tone for the sidebar/input, and a muted border
        // color for hairline separators.
        surface: {
          DEFAULT: "#0d0d0f",
          raised: "#171718",
          hover: "#212123",
          border: "#2a2a2c",
        },
        accent: {
          DEFAULT: "#4f8cff",
        },
      },
      borderRadius: {
        pill: "999px",
      },
    },
  },
  plugins: [],
};

export default config;
