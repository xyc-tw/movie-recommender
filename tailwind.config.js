/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./*.{html,js}", "./templates/*.html", "./static/*.js"],
  theme: {
    extend: {
      colors: {
        "color-1": "#36213e",
        "color-2": "#554971",
        "color-3": "#63768d",
        "color-4": "#8ac6d0",
        "color-5": "#b8f3ff",
      },
    },
    fontFamily: {
      Righteous: ["Righteous, sans-serif"],
      Montserrat: ["Montserrat, sans-serif"],
    },
  },
  plugins: [],
};
