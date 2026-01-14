import React, { useMemo } from "react";
import {
  Box,
  Paper,
  Stack,
  Typography,
  Button,
  Chip,
  Divider,
  ToggleButton,
  ToggleButtonGroup,
  IconButton,
} from "@mui/material";
import {
  Brightness4,
  Brightness7,
  Chat as ChatIcon,
  PhotoCamera as PhotoIcon,
  Send as SendIcon,
  RestartAlt,
  InfoOutlined,
} from "@mui/icons-material";
import type { PaletteMode } from "@mui/material";

export type Language = "en" | "si";

interface HomeProps {
  mode: PaletteMode;
  toggleMode: () => void;
  lang: Language;
  onLangChange: (lang: Language) => void;
  onStartChat: () => void;
}

const STRINGS: Record<Language, any> = {
  en: {
    title: "Paddy Disease Assistant",
    subtitle:
      "Identify common paddy leaf diseases by uploading images and describing symptoms. The assistant will suggest the most likely disease and provide management guidance.",
    howTitle: "How it works",
    step1Title: "1) Upload images (0–5)",
    step1Desc:
      "Upload up to 5 clear leaf images. Try to capture the lesion areas in focus with good lighting.",
    step2Title: "2) Describe symptoms",
    step2Desc:
      "Type key symptoms (spots, color changes, drying tips), crop stage (nursery/tillering/panicle), and recent weather.",
    step3Title: "3) Ask follow-up questions",
    step3Desc:
      "Ask about treatment, prevention, fertilizer guidance, spread conditions, or best practices for your situation.",
    detectTitle: "Diseases it can detect",
    detectNote:
      "Predictions are based on the uploaded images and model confidence. Always confirm in-field if symptoms are mixed.",
    tipsTitle: "Tips for best results",
    tip1: "Use sharp images (avoid blur) and include the affected area clearly.",
    tip2: "Upload multiple angles if available (2–4 images is ideal).",
    tip3: "Add crop stage + weather to improve recommendations.",
    tip4: "If the field has multiple symptoms, mention that in the text.",
    actionsTitle: "Get started",
    startChat: "Open chat",
    newChatHint:
      "You can switch language anytime. Changing language will start a new session in chat.",
    langEN: "EN",
    langSI: "සිංහල",
  },
  si: {
    title: "වී රෝග සහායකයා",
    subtitle:
      "පත්‍ර රූප උඩුගත කර ලක්ෂණ විස්තර කිරීම මඟින් වී පත්‍ර රෝග හඳුනාගන්න. සහායකයා වැඩි ඉඩකඩ ඇති රෝගය යෝජනා කර කළමනාකරණ උපදෙස් දෙනවා.",
    howTitle: "මෙය ක්‍රියා කරන ආකාරය",
    step1Title: "1) රූප උඩුගත කරන්න (0–5)",
    step1Desc:
      "පැහැදිලි පත්‍ර රූප 5ක් දක්වා උඩුගත කරන්න. ලප/ලක්ෂණ ඇති කොටස් හොඳින් පෙනෙන ලෙස, හොඳ ආලෝකයෙන් රූප ගන්න.",
    step2Title: "2) ලක්ෂණ විස්තර කරන්න",
    step2Desc:
      "ලප/වර්ණ වෙනස්වීම්/කෙළවර වියළීම වැනි ලක්ෂණ, වගා අදියර (නර්සරිය/ටිලරිං/පැනිකල්) සහ මෑත කාලගුණය සඳහන් කරන්න.",
    step3Title: "3) අමතර ප්‍රශ්න අහන්න",
    step3Desc:
      "ප්‍රතිකාර, වැළැක්වීම, පොහොර උපදෙස්, පැතිරීම සඳහා ඉතා හොඳ පරිසර තත්ත්ව, ඔබගේ තත්වයට ගැලපෙන හොඳම ක්‍රියාමාර්ග ගැන අහන්න.",
    detectTitle: "හඳුනාගත හැකි රෝග",
    detectNote:
      "අනුමානය රූප සහ මොඩල් විශ්වාසය මත පදනම් වේ. ලක්ෂණ මිශ්‍ර නම්, ක්ෂේත්‍රයේදීම තහවුරු කරන්න.",
    tipsTitle: "හොඳ ප්‍රතිඵල සඳහා උපදෙස්",
    tip1: "පැහැදිලි රූප භාවිතා කරන්න (බ්ලර් නොවිය යුතුය).",
    tip2: "හැකි නම් විවිධ කෝණ වලින් රූප කිහිපයක් උඩුගත කරන්න (2–4 හොඳයි).",
    tip3: "වගා අදියර + කාලගුණය එක් කළால் උපදෙස් වඩා හොඳ වේ.",
    tip4: "බහු ලක්ෂණ තිබේ නම්, ඒ බව වචන වලින්ද සඳහන් කරන්න.",
    actionsTitle: "ආරම්භ කරන්න",
    startChat: "චැට් එක විවෘත කරන්න",
    newChatHint:
      "භාෂාව ඕනෑම වේලාවක මාරු කළ හැක. චැට් එකේ භාෂාව මාරු කළ විට නව සැසියක් ආරම්භ වේ.",
    langEN: "EN",
    langSI: "සිංහල",
  },
};

const DISEASES = [
  { key: "blast", en: "Blast", si: "බ්ලෑස්ට්" },
  { key: "brown_spot", en: "Brown Spot", si: "දම් ලප රෝගය" },
  { key: "hispa", en: "Hispa", si: "හිස්පා කීට" },
  { key: "dead_heart", en: "Dead Heart", si: "මෘත හද" },
  { key: "tungro", en: "Tungro", si: "ටන්ග්රෝ" },
  { key: "normal", en: "Normal", si: "සාමාන්‍ය" },
];

const Home: React.FC<HomeProps> = ({
  mode,
  toggleMode,
  lang,
  onLangChange,
  onStartChat,
}) => {
  const t = useMemo(() => STRINGS[lang], [lang]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        p: { xs: 1.5, sm: 2 },
        backgroundImage:
          mode === "light"
            ? "linear-gradient(135deg, #e8f5e9, #e3f2fd)"
            : "linear-gradient(135deg, #020617, #111827)",
      }}
    >
      <Paper
        elevation={8}
        sx={{
          width: "100%",
          maxWidth: 980,
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <Box
          sx={{
            px: 2,
            py: 1.75,
            borderBottom: "1px solid",
            borderColor: "divider",
            display: "flex",
            justifyContent: "space-between",
            gap: 2,
            alignItems: "center",
          }}
        >
          <Stack spacing={0.5}>
            <Typography variant="h6" fontWeight={700}>
              {t.title}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {t.subtitle}
            </Typography>
          </Stack>

          <Stack direction="row" spacing={1} alignItems="center">
            <ToggleButtonGroup
              size="small"
              exclusive
              value={lang}
              onChange={(_, next) => {
                if (!next) return;
                onLangChange(next);
              }}
              aria-label="language"
            >
              <ToggleButton value="en" aria-label="English">
                {t.langEN}
              </ToggleButton>
              <ToggleButton value="si" aria-label="Sinhala">
                {t.langSI}
              </ToggleButton>
            </ToggleButtonGroup>

            <IconButton
              size="small"
              onClick={toggleMode}
              sx={{
                borderRadius: 999,
                bgcolor:
                  mode === "light"
                    ? "rgba(0,0,0,0.04)"
                    : "rgba(255,255,255,0.08)",
              }}
            >
              {mode === "light" ? (
                <Brightness4 fontSize="small" />
              ) : (
                <Brightness7 fontSize="small" />
              )}
            </IconButton>
          </Stack>
        </Box>

        {/* Body */}
        <Box sx={{ p: { xs: 2, sm: 2.5 } }}>
          <Stack spacing={2.25}>
            {/* How it works */}
            <Stack spacing={1}>
              <Stack direction="row" spacing={1} alignItems="center">
                <InfoOutlined fontSize="small" />
                <Typography variant="subtitle1" fontWeight={700}>
                  {t.howTitle}
                </Typography>
              </Stack>

              <Stack
                direction={{ xs: "column", md: "row" }}
                spacing={1.5}
                sx={{ mt: 0.5 }}
              >
                <Paper variant="outlined" sx={{ p: 1.5, flex: 1, borderRadius: 2 }}>
                  <Stack spacing={0.5}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <PhotoIcon fontSize="small" />
                      <Typography variant="subtitle2" fontWeight={700}>
                        {t.step1Title}
                      </Typography>
                    </Stack>
                    <Typography variant="body2" color="text.secondary">
                      {t.step1Desc}
                    </Typography>
                  </Stack>
                </Paper>

                <Paper variant="outlined" sx={{ p: 1.5, flex: 1, borderRadius: 2 }}>
                  <Stack spacing={0.5}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <SendIcon fontSize="small" />
                      <Typography variant="subtitle2" fontWeight={700}>
                        {t.step2Title}
                      </Typography>
                    </Stack>
                    <Typography variant="body2" color="text.secondary">
                      {t.step2Desc}
                    </Typography>
                  </Stack>
                </Paper>

                <Paper variant="outlined" sx={{ p: 1.5, flex: 1, borderRadius: 2 }}>
                  <Stack spacing={0.5}>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <ChatIcon fontSize="small" />
                      <Typography variant="subtitle2" fontWeight={700}>
                        {t.step3Title}
                      </Typography>
                    </Stack>
                    <Typography variant="body2" color="text.secondary">
                      {t.step3Desc}
                    </Typography>
                  </Stack>
                </Paper>
              </Stack>
            </Stack>

            <Divider />

            {/* Detectable diseases */}
            <Stack spacing={0.75}>
              <Typography variant="subtitle1" fontWeight={700}>
                {t.detectTitle}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {t.detectNote}
              </Typography>

              <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt: 0.5 }}>
                {DISEASES.map((d) => (
                  <Chip
                    key={d.key}
                    label={lang === "si" ? d.si : d.en}
                    variant="outlined"
                    sx={{ mb: 1 }}
                  />
                ))}
              </Stack>
            </Stack>

            <Divider />

            {/* Tips */}
            <Stack spacing={0.75}>
              <Typography variant="subtitle1" fontWeight={700}>
                {t.tipsTitle}
              </Typography>
              <Stack spacing={0.5}>
                <Typography variant="body2" color="text.secondary">
                  • {t.tip1}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • {t.tip2}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • {t.tip3}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • {t.tip4}
                </Typography>
              </Stack>
            </Stack>

            <Divider />

            {/* Actions */}
            <Stack
              direction={{ xs: "column", sm: "row" }}
              spacing={1.5}
              alignItems={{ xs: "stretch", sm: "center" }}
              justifyContent="space-between"
            >
              <Stack spacing={0.25}>
                <Typography variant="subtitle1" fontWeight={700}>
                  {t.actionsTitle}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {t.newChatHint}
                </Typography>
              </Stack>

              <Button
                variant="contained"
                size="large"
                onClick={onStartChat}
                startIcon={<ChatIcon />}
                sx={{ px: 3 }}
              >
                {t.startChat}
              </Button>
            </Stack>
          </Stack>
        </Box>
      </Paper>
    </Box>
  );
};

export default Home;
