import React, { useMemo, useState } from "react";
import {
  createTheme,
  CssBaseline,
  ThemeProvider,
} from "@mui/material";
import type { PaletteMode } from "@mui/material";  // ⬅ type-only import
import PaddyChat from "./components/PaddyChat";

const App: React.FC = () => {
  const [mode, setMode] = useState<PaletteMode>("light");

  const theme = useMemo(
    () =>
      createTheme({
        palette: {
          mode,
          primary: {
            main: "#2e7d32",
          },
          background: {
            default: mode === "light" ? "#e8f5e9" : "#020617",
            paper: mode === "light" ? "#ffffff" : "#0b1220",
          },
        },
        shape: {
          borderRadius: 16,
        },
      }),
    [mode]
  );

  const toggleMode = () =>
    setMode((prev) => (prev === "light" ? "dark" : "light"));

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <PaddyChat mode={mode} toggleMode={toggleMode} />
    </ThemeProvider>
  );
};

export default App;
