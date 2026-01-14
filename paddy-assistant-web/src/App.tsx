import React, { useMemo, useState } from "react";
import { createTheme, CssBaseline, ThemeProvider } from "@mui/material";
import type { PaletteMode } from "@mui/material";
import PaddyChat from "./components/PaddyChat";
import Home, { type Language } from "./Home";

type Page = "home" | "chat";

const App: React.FC = () => {
  const [mode, setMode] = useState<PaletteMode>("light");
  const [page, setPage] = useState<Page>("home");
  const [lang, setLang] = useState<Language>("en");

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: { main: "#2e7d32" },
          background: {
            default: mode === "light" ? "#e8f5e9" : "#020617",
            paper: mode === "light" ? "#ffffff" : "#0b1220",
          },
        },
        shape: { borderRadius: 16 },
      }),
    [mode]
  );

  const toggleMode = () => setMode((prev) => (prev === "light" ? "dark" : "light"));

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {page === "home" ? (
        <Home
          mode={mode}
          toggleMode={toggleMode}
          lang={lang}
          onLangChange={setLang}
          onStartChat={() => setPage("chat")}
        />
      ) : (
        <PaddyChat
          mode={mode}
          toggleMode={toggleMode}
          lang={lang}
          onLangChange={setLang}
          onBackHome={() => setPage("home")}
        />
      )}
    </ThemeProvider>
  );
};

export default App;
